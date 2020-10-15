import argparse
import os
import sys
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

import numpy as np

from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

import utils
import models
from lstm import lstm
import vgg_64
import resnet_64
from moving_mnist import MovingMNIST
from moving_dsprites import MovingDSprites
from mpi3d_toy import Mpi3dReal
import hickle
from metrics.metrics import _lpips_wrapper as lpips_loss

parser = argparse.ArgumentParser()
parser.add_argument("--ae_checkpoint", required=True, type = str, help = "Auto-encoder checkpoint")
parser.add_argument("--lstm_checkpoint", required=True, type = str, help = "LSTM checkpoint")
parser.add_argument("--input_frames", default=5, type=int, help = "Number of past frames")
parser.add_argument("--target_frames", default=30, type=int, help = "Number of frames to predict")
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
parser.add_argument("--num_samples", default = 1000, type = int)
parser.add_argument("--output_file_name", default = "ssim_psnr.hkl", type = str)


args = parser.parse_args()

lpips_path = "pretrained_models/alex.pth"

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

Ec.load_state_dict(ae_checkpoint["content_encoder"])
Ep.load_state_dict(ae_checkpoint["position_encoder"])
D.load_state_dict(ae_checkpoint["decoder"])
lstm_model.load_state_dict(lstm_checkpoint["lstm"])

Ec.eval()
Ep.eval()
D.eval()
lstm_model.eval()

psnr_all = np.zeros([args.num_samples, args.target_frames])
ssim_all = np.zeros([args.num_samples, args.target_frames])
lpips_all = np.zeros([args.num_samples, args.target_frames])

for k in range(int(args.num_samples/args.batch_size)):
    x = next(data_generator).cuda()
    x.transpose_(0,1)

    pred_frames = []
    pred_z_p_list = []
    z_c = Ec(x[args.input_frames-1])
    lstm_hidden = lstm_model.init_hidden()

    for i in range(args.input_frames + args.target_frames-1):
        if i < args.input_frames:
            z_p = Ep(x[i])
        else:
            z_p = pred_z_p_list[-1]
        
        with torch.no_grad():
            pred_z_p, lstm_hidden = lstm_model(torch.cat([z_p,z_c[0]],1), lstm_hidden)
            pred_frame = D([z_c,pred_z_p])
        pred_z_p_list.append(pred_z_p)

        if i >= (args.input_frames-1):
            pred_frames.append(pred_frame)

    pred_frames = torch.stack(pred_frames,0).contiguous()
    x = x[args.input_frames:].contiguous()
    lpips = lpips_loss(pred_frames,x,lpips_path)
    lpips_all[k*args.batch_size:(k+1)*args.batch_size,:] = lpips.transpose(0,1).detach().cpu().numpy()

    pred_frames = pred_frames.transpose(0,1).transpose(2,3).transpose(3,4).detach().cpu().numpy()
    x = x.transpose(0,1).transpose(2,3).transpose(3,4).detach().cpu().numpy()

    for i in range(args.batch_size):
        target_video = (x[i]*255).astype(np.uint8)
        pred_video = (pred_frames[i]*255).astype(np.uint8)
        for j in range(pred_frames.shape[1]):
            psnr_all[k*args.batch_size+i,j] = psnr(target_video[j],pred_video[j])
            ssim_all[k*args.batch_size+i,j] = ssim(target_video[j],pred_video[j], multichannel=True)

print("mean LPIPS distance:",lpips_all.mean(0))
print("mean SSIM metric:", ssim_all.mean(0))
print("mean PSNR metric:",psnr_all.mean(0))
hickle.dump({"psnr":psnr_all,"ssim":ssim_all,"lpips":lpips_all}, os.path.join(args.log_dir,args.output_file_name))

