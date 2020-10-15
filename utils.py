import sys
import itertools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

def log_gradients(module, summary_writer, global_step = None):
    for tag, value in module.named_parameters():
        tag = tag.replace(".","/")
        if value.grad is not None:
            summary_writer.add_scalar(tag+"/grad_norm", value.grad.norm(), global_step = global_step)
            summary_writer.add_histogram(tag+"/grad", value.grad, global_step = global_step)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def shuffle_dim(variables, axis = 0):
    perm_variables =[]
    for variable in variables:
        B = variable.shape[0]
        perm = torch.randperm(B)
        perm_variables.append(variable[perm])
    return perm_variables

def shuffle_time(variables):
    B = variables.shape[0]

    perm_variables = []
    for z_t in variables.split(1,1):
        perm = torch.randperm(B)
        perm_z_t = z_t[perm]
        perm_variables.append(perm_z_t)

    return torch.cat(perm_variables, 1)

def save_gif(gif_fname, images, fps):
    """ 
    To generate a gif from image files, first generate palette from images
    and then generate the gif from the images and the palette.
    ffmpeg -i input_%02d.jpg -vf palettegen -y palette.png
    ffmpeg -i input_%02d.jpg -i palette.png -lavfi paletteuse -y output.gif

    Alternatively, use a filter to map the input images to both the palette
    and gif commands, while also passing the palette to the gif command.
    ffmpeg -i input_%02d.jpg -filter_complex "[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse" -y output.gif

    To directly pass in numpy images, use rawvideo format and `-i -` option.
    """
    from subprocess import Popen, PIPE
    head, tail = os.path.split(gif_fname)
    if head and not os.path.exists(head):
        os.makedirs(head)
    h, w, c = images[0].shape
    cmd = ['ffmpeg', '-y',
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-r', '%.02f' % fps,
           '-s', '%dx%d' % (w, h), 
           '-pix_fmt', {1: 'gray', 3: 'rgb24', 4: 'rgba'}[c],
           '-i', '-',
           '-filter_complex', '[0:v]split[x][z];[z]palettegen[y];[x][y]paletteuse',
           '-r', '%.02f' % fps,
           '%s' % gif_fname]
    proc = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    for image in images:
        proc.stdin.write(image.tostring())
    out, err = proc.communicate()
    if proc.returncode:
        err = '\n'.join([' '.join(cmd), err.decode('utf8')])
        raise IOError(err)
    del proc
