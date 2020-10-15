import argparse
import os
import sys

import torch
import torch.nn as nn

import numpy as np

import utils
from lstm import lstm
import models
from moving_mnist import MovingMNIST
from moving_dsprites import MovingDSprites
from mpi3d_toy import Mpi3dReal
import vgg_64
import resnet_64
import hickle

parser = argparse.ArgumentParser()
parser.add_argument("--ae_checkpoint", required=True, type = str, help = "Auto-encoder checkpoint")
parser.add_argument("--lstm_checkpoint", required=True, type = str, help = "LSTM checkpoint")
parser.add_argument("--input_frames", default=5, type=int, help = "Number of past frames")
parser.add_argument("--target_frames", default=10, type=int, help = "Number of frames to predict")
parser.add_argument("--log_dir", default="logs", type=str, help = "Directory to store results in")
parser.add_argument("--z_dims", default=5, type=int, help = "Number of pose dims")
parser.add_argument("--g_dims", default=128, type=int, help = "Number of content dims")
parser.add_argument("--rnn_size", default=256, type = int, help = "LSTM hidden state size")
parser.add_argument("--rnn_layers", default=2, type = int, help = "num LSTM layers")
parser.add_argument("--num_channels", default=3, type=int, help = "Number of color channles")
parser.add_argument("--color", dest="color", action="store_true", help = "generate greyscale dataset") #Only for mnist and dsprites dataset
parser.add_argument("--no_color", dest="color", action="store_false", help = "generate color dataset")
parser.set_defaults(color=True)
parser.add_argument("--skips", dest = "skips", action = "store_true", help = "Use encoder decoder skips")
parser.add_argument("--no_skips", dest = "skips", action="store_false", help = "Dont use encoder decoder skips")
parser.set_defaults(skips=False)
parser.add_argument("--batch_size", default=100, type=int, help = "Batch size")
parser.add_argument("--data_root", default="data", type=str, help = "Directory with dataset")
parser.add_argument("--dataset", default = "mnist", type = str, choices = ["mnist","dsprites","mpi3d_real"], help = "Dataset")
parser.add_argument("--rotate_sprites", dest = "rotate_sprites", action="store_true", help = "Rotate sprites in a sequence") #Only for dsprites dataset
parser.add_argument("--no_rotate_sprites", dest = "rotate_sprites", action="store_false", help = "Do not rotate sprites in a sequence")
parser.set_defaults(rotate_sprites=False)#Used only for dsprites dataset

args = parser.parse_args()

if args.dataset == "mnist":
    data = MovingMNIST(False, args.data_root, color = args.color, seq_len = args.input_frames + args.target_frames, deterministic = True)
elif args.dataset == "dsprites":
    data = MovingDSprites(False, args.data_root, color = args.color, seq_len = args.input_frames + args.target_frames,rotate = args.rotate_sprites, deterministic = True)
else:
    data = Mpi3dReal(False, args.data_root, seq_len = args.input_frames + args.target_frames, deterministic = True)

data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=5, drop_last=True)

def get_data_batch():
    while True:
        for seq in data_loader:
            seq.transpose_(3,4).transpose_(2,3)
            yield seq
data_generator = get_data_batch()

if args.dataset != "mpi3d_real":
    Ec = models.content_encoder(args.g_dims, nc = args.num_channels).cuda()
    Ep = models.pose_encoder(args.z_dims, nc = args.num_channels).cuda()
    D = models.decoder(args.g_dims, args.z_dims, nc = args.num_channels, skips = args.skips).cuda()
else:
    Ec = vgg_64.encoder(args.g_dims, nc = args.num_channels).cuda()
    Ep = resnet_64.pose_encoder(args.z_dims, nc = args.num_channels).cuda()
    D = vgg_64.drnet_decoder(args.g_dims, args.z_dims, nc = args.num_channels).cuda()

lstm_model = lstm(args.g_dims + args.z_dims, args.z_dims, args.rnn_size, args.rnn_layers, args.batch_size).cuda()

ae_checkpoint = torch.load(args.ae_checkpoint)
lstm_checkpoint = torch.load(args.lstm_checkpoint)

x = next(data_generator).cuda()
x.transpose_(0,1)

Ec.load_state_dict(ae_checkpoint["content_encoder"])
Ep.load_state_dict(ae_checkpoint["position_encoder"])
D.load_state_dict(ae_checkpoint["decoder"])
lstm_model.load_state_dict(lstm_checkpoint["lstm"])

Ec.eval()
Ep.eval()
D.eval()
lstm_model.eval()

pred_frames = []
pred_h_p_list = []
h_c = Ec(x[args.input_frames-1])
lstm_hidden = lstm_model.init_hidden()
for i in range(args.input_frames + args.target_frames):
    if i < args.input_frames:
        h_p = Ep(x[i])
    else:
        h_p = pred_h_p_list[-1]
    
    pred_h_p, lstm_hidden = lstm_model(torch.cat([h_p,h_c[0]],1), lstm_hidden)
    pred_frame = D([h_c,pred_h_p])
    pred_h_p_list.append(pred_h_p)
    if i >= args.input_frames:
        pred_frames.append(pred_frame)

pred_frames = torch.stack(pred_frames, 1)
pred_frames = pred_frames.transpose(2,3).transpose(3,4).detach().cpu().numpy()

x = x.transpose(0,1).transpose(2,3).transpose(3,4).detach().cpu().numpy()

os.makedirs(args.log_dir, exist_ok=True)

for i in range(args.batch_size):
    input_frames = (x[i, :args.input_frames]*255).astype(np.uint8)
    target_frames = (x[i, args.input_frames:]*255).astype(np.uint8)
    predicted_frames = (pred_frames[i]*255).astype(np.uint8)

    utils.save_gif(os.path.join(args.log_dir,"input_frames_batch_"+str(i)+".gif"), input_frames, 5)
    utils.save_gif(os.path.join(args.log_dir,"ground_truth_frames_batch_"+str(i)+".gif"), target_frames, 5)
    utils.save_gif(os.path.join(args.log_dir,"predicted_frames_batch_"+str(i)+".gif"), predicted_frames, 5)
