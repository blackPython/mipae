import argparse
import os
import sys

import torch
import torch.nn as nn

import numpy as np

import imageio

import utils
from lstm import lstm
import models
from moving_mnist import MovingMNIST
from moving_dsprites import MovingDSprites
from mpi3d_toy import Mpi3dReal
import vgg_64
import resnet_64

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, type = str, help = "auto-encoder checkpoint")
parser.add_argument("--log_dir", default="logs", type = str, help = "Directory to store results")
parser.add_argument("--z_dims", default=5, type=int, help = "Number of pose dims")
parser.add_argument("--g_dims", default=128, type=int, help = "Number of content dims")
parser.add_argument("--num_channels", default=3, type=int, help = "Number of color channels")
parser.add_argument("--color", dest="color", action="store_true", help = "generate greyscale dataset") #Only for mnist and dsprites dataset
parser.add_argument("--no_color", dest="color", action="store_false", help = "generate color dataset")
parser.set_defaults(color=True)
parser.add_argument("--skips", dest = "skips", action = "store_true", help = "Use encoder decoder skips")
parser.add_argument("--no_skips", dest = "skips", action="store_false", help = "Dont use encoder decoder skips")
parser.set_defaults(skips=False)
parser.add_argument("--batch_size", default=100, type=int, help = "Batch size")
parser.add_argument("--data_root", default="data", type=str, help = "Directory with dataset")
parser.add_argument("--examples", default= 10, type=int, help = "Number of content frames per image")
parser.add_argument("--num_samples", default = 25, type=int, help = "Number of samples")
parser.add_argument("--time_step", default=4, type=int)
parser.add_argument("--dataset", default = "mnist", type = str, choices = ["mnist","dsprites","mpi3d_real"], help = "Dataset")
parser.add_argument("--rotate_sprites", dest = "rotate_sprites", action="store_true", help = "Rotate sprites in a sequence") #Only for dsprites dataset
parser.add_argument("--no_rotate_sprites", dest = "rotate_sprites", action="store_false", help = "Do not rotate sprites in a sequence")
parser.set_defaults(rotate_sprites=False)#Used only for dsprites dataset

args = parser.parse_args()

if args.dataset == "mnist":
    data = MovingMNIST(False, args.data_root, color = args.color)
elif args.dataset == "dsprites":
    data = MovingDSprites(False, args.data_root, color= args.color , rotate=args.rotate_sprites, deterministic=True)
else:
    data = Mpi3dReal(False, args.data_root, deterministic = True)
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

checkpoint = torch.load(args.checkpoint)

Ec.load_state_dict(checkpoint["content_encoder"])
Ep.load_state_dict(checkpoint["position_encoder"])
D.load_state_dict(checkpoint["decoder"])
Ec.eval()
Ep.eval()
D.eval()

results = []

for i in range(args.num_samples):
    x_s = next(data_generator).cuda()[:10].transpose(0,1)
    x_target = next(data_generator).cuda()[:args.examples].transpose(0,1)
    x_source = x_s[:,0][:,None]
    h_c = Ec(x_target[0])
    position_list = [Ep(x_s[i])[0][None] for i in range(1,20, args.time_step)]

    generated_images = torch.stack([D([h_c, torch.cat([h_p]*args.examples,0)]) for h_p in position_list], dim = 1)
    generated_images = torch.cat(generated_images.split(1,0), 3)
    generated_images = torch.cat(generated_images.split(1,1), 4).squeeze(0).squeeze(0)

    source_images = x_source[1:20:args.time_step, 0]
    source_images = torch.cat(source_images.split(1,0), 3).squeeze(0)

    content_images = x_target[0]
    content_images = torch.cat(content_images.split(1,0), 2).squeeze(0)

    result_image = torch.cat([content_images, generated_images], dim = 2)
    source_images = torch.cat([torch.zeros_like(x_source[0,0]), source_images], dim = 2)
    result_image = torch.cat([source_images, result_image], dim = 1).transpose(0,1).transpose(1,2)

    results.append(result_image.detach().cpu().numpy())

os.makedirs(args.log_dir, exist_ok=True)

for i,result in enumerate(results):
    result = (result*255).astype(np.uint8)
    imageio.imwrite(os.path.join(args.log_dir, "image_"+str(i)+".png"), result)
