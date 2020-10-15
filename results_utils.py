import numpy as np
import torch

def pad_const(x, const = 1, thickness = 2,axis = None):
    if not axis:
        axis = list(range(len(x.shape)))

    for i in axis:
        curr_shape = x.shape
        padding = torch.ones(curr_shape[0:i] + (thickness,) + curr_shape[(i+1):], dtype = x.dtype).cuda()*const
        x = torch.cat([padding,x,padding], dim = i)
    
    return x

