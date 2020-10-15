import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

###
#Emily losses are not MI estimates added them here for convinence
###
def discriminator_loss(out_true,out_false):
    out_true, out_false = F.sigmoid(out_true), F.sigmoid(out_false)
    target_ones, target_zeros = torch.ones_like(out_true), torch.zeros_like(out_false)
    return 0.5 * F.binary_cross_entropy(out_true,target_ones) + \
        0.5 * F.binary_cross_entropy(out_false, target_zeros)

def emily_sd_loss(x):
    target = torch.ones_like(x) * 0.5
    return F.binary_cross_entropy(torch.sigmoid(x), target)


####################
# MI LOWER BOUNDS ##
####################
def donsker_vardhan_lower_bound(out_true,out_false):
    return out_true.mean() - out_false.logsumexp(0) + math.log(out_false.shape[0])

def tuba_lower_bound(out_true,out_false, log_baseline = None):
    if log_baseline is not None:
        out_true = out_true - log_baseline
        out_false = out_false - log_baseline
    joint_term = out_true.mean()
    marg_term = out_false.logsumexp(0).exp()/out_false.shape[0]
    return joint_term - marg_term + 1.0

def nwj_lower_bound(out_true,out_false):
    return tuba_lower_bound(out_true,out_false,1.0)

def js_fgan_lower_bound(out_true,out_false):
    return -1*F.softplus(-1*out_true).mean() - F.softplus(out_false).mean()

def js_mi_lower_bound(out_true,out_false):
    return nwj_lower_bound(out_true + 1, out_false + 1)

def smile_lower_bound(out_true,out_false, clamp = 5.0):
    out_true = torch.clamp(out_true, -1*clamp, clamp)
    out_false = torch.clamp(out_false, -1*clamp, clamp)
    return js_fgan_lower_bound(out_true,out_false)

def smile_mi_lower_bound(out_true,out_false, clamp = 5.0):
    out_true = torch.clamp(out_true, -1*clamp, clamp)
    out_false = torch.clamp(out_false, -1*clamp, clamp)
    return donsker_vardhan_lower_bound(out_true, out_false)