import os

import torch
import torch.nn.functional as F

from .lpips.loss import PerceptualLoss
from .ssim import ssim_loss

lpips_model = None

def psnr(prediction,target):
    """ Assuming inputs are 5D tensors [T,B,NC,H,W] """
    mse = F.mse_loss(prediction,target, reduction = "none").mean(dim = [2,3,4])
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

def _ssim_wrapper(sample, gt):
    nt, bsz = sample.shape[0], sample.shape[1]
    img_shape = sample.shape[2:]
    ssim = ssim_loss(sample.view(nt * bsz, *img_shape), gt.view(nt * bsz, *img_shape), max_val=1., reduction='none')
    return ssim.mean(dim=[2, 3]).view(nt, bsz, img_shape[0])

def _lpips_wrapper(sample, gt, lpips_path):
    global lpips_model
    if lpips_model is None:
        lpips_model = PerceptualLoss(lpips_path)
    nt, bsz = sample.shape[0], sample.shape[1]
    img_shape = sample.shape[2:]
    if img_shape[0] == 1:
        sample_ = sample.repeat(1, 1, 3, 1, 1)
        gt_ = gt.repeat(1, 1, 3, 1, 1)
    else:
        sample_ = sample
        gt_ = gt
    lpips = lpips_model(sample_.view(nt * bsz, 3, *img_shape[1:]), gt_.view(nt * bsz, 3, *img_shape[1:]))
    return lpips.view(nt, bsz)
