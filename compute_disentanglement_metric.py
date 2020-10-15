import argparse
import os
import sys
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np

import utils
import models
import vgg_64
import resnet_64
from disentanglement_metric import mig_metric
from static_dsprites import StaticDSprites
from static_mpi3d_toy import Mpi3dReal

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, type = str)
parser.add_argument("--g_dims", default=128, type = int)
parser.add_argument("--z_dims", default=5, type = int)
parser.add_argument("--num_channels", default=3, type=int)
parser.add_argument("--color", dest="color", action="store_true")
parser.add_argument("--no_color", dest="color", action="store_false")
parser.set_defaults(color=True)
parser.add_argument("--skips", dest = "skips", action = "store_true")
parser.add_argument("--no_skips", dest = "skips", action="store_false")
parser.set_defaults(skips=False)
parser.add_argument("--batch_size", default = 10, type = int)
parser.add_argument("--data_root", default = "data", type = str)
parser.add_argument("--dataset", default="dsprites", choices = ["dsprites","mpi3d_real"], type = str)
parser.add_argument("--rotate_sprites", dest="rotate_sprites", action = "store_true")
parser.add_argument("--no_rotate_sprites", dest="rotate_sprites", action = "store_false")
parser.set_defaults(rotate_sprites=False)
parser.add_argument("--samples", default = 100000, type = int)

args = parser.parse_args()

if args.dataset == "dsprites":
    data = StaticDSprites(args.data_root, color = args.color, return_factors = True)
elif args.dataset == "mpi3d_real":
    data = Mpi3dReal(args.data_root, return_factors = True)

data_loader =  DataLoader(data, batch_size = args.batch_size, shuffle = True, num_workers = 5, drop_last = True)

def get_data_batch():
    while  True:
        for seq in data_loader:
            seq[1].transpose_(2,3).transpose_(1,2)
            yield seq
data_generator = get_data_batch()

if args.dataset == "dsprites":
    Ec = models.content_encoder(args.g_dims, nc = args.num_channels).cuda()
    Ep = models.pose_encoder(args.z_dims, nc = args.num_channels).cuda()
elif args.dataset == "mpi3d_real":
    Ec = vgg_64.encoder(args.g_dims, nc = args.num_channels).cuda()
    Ep = resnet_64.pose_encoder(args.z_dims, nc = args.num_channels).cuda()

checkpoint = torch.load(args.checkpoint)

Ec.load_state_dict(checkpoint["content_encoder"])
Ep.load_state_dict(checkpoint["position_encoder"])
Ec.eval()
Ep.eval()

latent_c = None
latent_p = None
factors_p = None
factors_c = None
for i in range(math.ceil(args.samples/args.batch_size)):
    f, x = next(data_generator)
    f_c, f_p = f
    x = x.cuda()
    if factors_c is None:
        factors_c = f_c.detach().cpu().numpy()
        factors_p = f_p.detach().cpu().numpy()
    else:
        factors_c = np.vstack([factors_c, f_c.detach().cpu().numpy()])
        factors_p = np.vstack([factors_p, f_p.detach().cpu().numpy()])

    z_c = Ec(x)[0].detach().cpu().numpy()
    z_p = Ep(x).detach().cpu().numpy()

    if latent_c is None:
        latent_c = z_c
        latent_p = z_p
    else:
        latent_c = np.vstack([latent_c,z_c])
        latent_p = np.vstack([latent_p,z_p])

score, mis = mig_metric((latent_c,latent_p),(factors_p,factors_c))

print("Mutual Information Gap:",score)
print("I(f_c,z_c):", mis[0])
print("I(f_c,z_p):", mis[1])
print("I(f_p,z_c):", mis[2])
print("I(f_p,z_p):", mis[3])
