import os
import glob
import math
import sys
import itertools
import random

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import models
import resnet_64
import resnet_128
import vgg_128
import vgg_64
from lstm import lstm
from moving_mnist import MovingMNIST
from moving_dsprites import MovingDSprites
from mpi3d_toy import Mpi3dReal
import utils
import critics
import mi_estimators

class Solver(object):
    def __init__(self, nets, optims, args, estimators = {},extra_keys = {}):
        torch.set_default_dtype(torch.float32)
        self.cpu = torch.device('cpu')

        args = args.__dict__
        for key in args:
            setattr(self, key, args[key])

        self.nets = nets
        self.estimators = estimators
        self.optims = optims
        self.extra_keys = extra_keys

        self.load_checkpoint_or_initialize(extra_keys)

        for name in self.nets:
           self.nets[name] = nn.DataParallel(self.nets[name]) 

        if self.dataset == "mnist":
            train_data = MovingMNIST(True, self.data_root, seq_len = self.input_frames + self.target_frames, color = self.color, deterministic = self.deterministic)
            test_data = MovingMNIST(False, self.data_root, seq_len = self.input_frames + self.target_frames, color = self.color, deterministic = self.deterministic)
        elif self.dataset == "dsprites":
            train_data = MovingDSprites(True, self.data_root, seq_len = self.input_frames + self.target_frames, color = self.color, rotate = self.rotate_sprites, deterministic = self.deterministic)
            test_data = MovingDSprites(False, self.data_root, seq_len = self.input_frames + self.target_frames, color = self.color, rotate = self.rotate_sprites, deterministic = self.deterministic)
        elif self.dataset == "mpi3d_real":
            train_data = Mpi3dReal(True, self.data_root, seq_len = self.input_frames + self.target_frames, deterministic = self.deterministic)
            test_data = Mpi3dReal(False, self.data_root, seq_len = self.input_frames + self.target_frames, deterministic = self.deterministic)
        else:
            raise NotImplementedError()

        self.dataset_len = len(train_data)

        train_loader = DataLoader(train_data, batch_size = self.batch_size, shuffle = True, num_workers = 5, drop_last=True)
        test_loader = DataLoader(test_data, batch_size = self.batch_size, shuffle = True, num_workers = 5, drop_last=True)

        def get_training_batch():
            while True:
                for sequence in train_loader:
                    sequence.transpose_(3,4).transpose_(2,3)
                    yield sequence
        
        def get_test_batch():
            while True:
                for sequence in test_loader:
                    sequence.transpose_(3,4).transpose_(2,3)
                    yield sequence
        
        self.train_generator = get_training_batch()
        self.test_generator = get_test_batch()

        eval_dir = os.path.join(self.log_dir, "eval")

        self.train_summary_writer = SummaryWriter(log_dir = self.log_dir)
        self.test_summary_writer = SummaryWriter(log_dir = eval_dir)
        
        #Writing hyperparameters summary
        for name in args:
            self.train_summary_writer.add_text("Hyperparameters/"+name, str(args[name]))

    def set_mode(self, mode):
        if mode == "train":
            for net in self.nets:
                self.nets[net].train()
        else:
            for net in self.nets:
                self.nets[net].eval()


    def load_checkpoint_or_initialize(self, extra_keys):
        # Here the extra_keys should be a dict (containing default values)
        chkp_files = sorted(glob.glob(self.log_dir+"/"+self.name+r"-*.pth"), key = os.path.getmtime, reverse = True)

        checkpoint = None
        if chkp_files:
            checkpoint = torch.load(chkp_files[0], map_location= self.cpu)

        if checkpoint:
            for name in self.nets:
                self.nets[name].load_state_dict(checkpoint[name])
            
            for name in self.estimators:
                self.estimators[name].load_state_dict(checkpoint[name])
            for name in self.optims:
                self.optims[name].load_state_dict(checkpoint[name])

            for name in extra_keys:
                setattr(self, name, checkpoint[name])
            
            self.global_itr = checkpoint["global_itr"]
        
        else:
            for name in extra_keys:
                setattr(self, name, extra_keys[name])
            for name in self.nets:
                self.nets[name].apply(utils.init_weights)
            self.global_itr = 0


    def save_checkpoint(self, extra_keys = []):
        checkpoint = {"global_itr" : self.global_itr}
        for name in self.nets:
            checkpoint[name] = self.nets[name].module.state_dict()
        for name in self.estimators:
            checkpoint[name] = self.estimators[name].state_dict()
        for name in self.optims:
            checkpoint[name] = self.optims[name].state_dict()
        for name in extra_keys:
            checkpoint[name] = getattr(self,name)

        chkp_files = sorted(glob.glob(self.log_dir + "/"+self.name+r"-*.pth"), key = os.path.getmtime, reverse = True)
        if len(chkp_files) == self.max_checkpoints:
            os.remove(chkp_files[-1])
        chkp_path = self.log_dir + "/"+self.name + "-" + str(self.global_itr) + ".pth"
        
        torch.save(checkpoint, chkp_path)

    def train(self):
        
        while self.global_itr < self.niters:
            self.set_mode("train")
            for i in tqdm(range(self.epoch_size), desc = "[" + str(self.global_itr)+"/"+str(self.niters)+"]"):
                self.train_step(summary = (i==0 and self.global_itr%self.summary_freq == 0))
            
            self.set_mode("eval")
            self.eval_step()

            if self.global_itr%self.checkpoint_freq == 0:
                self.save_checkpoint(self.extra_keys)
            
            self.global_itr += 1

        self.save_checkpoint(self.extra_keys)

    def train_step(self,summary = False):
        raise NotImplementedError()

    def eval_step(self):
        raise NotImplementedError()

class SolverAutoencoder(Solver):
    def __init__(self, args):
        args.deterministic = True
        if args.dataset in ["mnist","dsprites"]:
            content_encoder = models.content_encoder(args.g_dims, nc = args.num_channels).cuda()
            position_encoder = models.pose_encoder(args.z_dims, nc = args.num_channels,normalize= args.normalize_position).cuda()
        else:
            content_encoder = vgg_64.encoder(args.g_dims, nc = args.num_channels).cuda()
            position_encoder = resnet_64.pose_encoder(args.z_dims, nc = args.num_channels).cuda()

        if args.dataset == "mpi3d_real":
            decoder = vgg_64.drnet_decoder(args.g_dims, args.z_dims, nc = args.num_channels).cuda()
        else:
            decoder = models.decoder(args.g_dims, args.z_dims, nc = args.num_channels, skips = args.skips).cuda()
        
        self.content_frames = 1
        if args.content_lstm:
            content_encoder = models.content_encoder_lstm(args.g_dims, content_encoder, args.batch_size)
            self.content_frames = args.input_frames

        discriminator = models.scene_discriminator(args.z_dims).cuda()
        nets = {
            "content_encoder" : content_encoder,
            "position_encoder" : position_encoder,
            "decoder" : decoder,
            "discriminator" : discriminator,
        }

        self.encoder_decoder_parameters = itertools.chain(*[
            content_encoder.parameters(),
            position_encoder.parameters(),
            decoder.parameters(),
        ])

        encoder_decoder_optim = torch.optim.Adam(
            self.encoder_decoder_parameters,
            lr = args.lr,
            betas = (args.beta1, 0.999)
        )

        discriminator_optim = torch.optim.Adam(
            discriminator.parameters(),
            lr = args.lr,
            betas = (args.beta1, 0.999)
        )

        optims = {
            "encoder_decoder_optim" : encoder_decoder_optim,
            "discriminator_optim" : discriminator_optim,
        }
        
        super().__init__(nets, optims, args)

    def train_step(self, summary = False):
        Ec = self.nets["content_encoder"]
        Ep = self.nets["position_encoder"]
        D = self.nets["decoder"]
        C = self.nets["discriminator"]

        encoder_decoder_optim = self.optims["encoder_decoder_optim"]
        discriminator_optim = self.optims["discriminator_optim"]

        #Train discriminator
        x = next(self.train_generator).cuda().transpose(0,1)
        
        h_p1 = Ep(x[random.randint(0, self.input_frames + self.target_frames-1)]).detach()
        h_p2 = Ep(x[random.randint(0, self.input_frames + self.target_frames-1)]).detach()

        rp = torch.randperm(self.batch_size).cuda()
        h_p2_perm = h_p2[rp]

        out_true = C([h_p1, h_p2])
        out_false = C([h_p1, h_p2_perm])

        if self.sd_loss == "emily":
            disc_loss = mi_estimators.discriminator_loss(out_true,out_false)
        elif self.sd_loss == "js":
            disc_loss = -1*mi_estimators.js_fgan_lower_bound(out_true,out_false)
        elif self.sd_loss == "smile":
            disc_loss = -1*mi_estimators.smile_lower_bound(out_true,out_false)
        else:
            raise NotImplementedError()

        discriminator_optim.zero_grad()
        disc_loss.backward()
        
        if summary:
            utils.log_gradients(C, self.train_summary_writer, global_step = self.global_itr)
        
        discriminator_optim.step()
        if summary:
            self.train_summary_writer.add_scalar("discriminator_loss", disc_loss, global_step = self.global_itr)

        k = random.randint(self.content_frames,self.input_frames + self.target_frames-self.content_frames)

        x = next(self.train_generator).cuda().transpose(0,1)
        if self.dataset != "lpc":
            x_c1 = x[0:self.content_frames].squeeze(0)
            x_c2 = x[k:(k+self.content_frames)].squeeze(0)
        else:
            x_c1 = x[k:(k+self.content_frames)].squeeze(0)
            x_c2 = x[0:self.content_frames].squeeze(0)
            
        x_p1 = x[k]
        x_p2 = x[random.randint(0, self.input_frames + self.target_frames-1)]

        h_content, skips = Ec(x_c1)
        h_content_1 = Ec(x_c2)[0].detach()
        h_position = Ep(x_p1)
        h_position_1 = Ep(x_p2).detach()

        sim_loss = F.mse_loss(h_content, h_content_1)

        x_rec = D([[h_content,skips], h_position])
        if self.recon_loss_type == "mse":
            rec_loss = F.mse_loss(x_rec, x_p1)
        elif self.recon_loss_type == "l1":
            rec_loss = F.l1_loss(x_rec, x_p1)

        if self.sd_loss == "emily":
            out = C([h_position, h_position_1])
            sd_loss = mi_estimators.emily_sd_loss(out)
        else:
            rp = torch.randperm(self.batch_size).cuda()
            h_p2_perm = h_position_1[rp]

            out_true = C([h_position, h_position_1])
            out_false = C([h_position, h_p2_perm])

            if self.sd_loss == "js":
                sd_loss = mi_estimators.js_mi_lower_bound(out_true,out_false)
            elif self.sd_loss == "smile":
                sd_loss = mi_estimators.smile_mi_lower_bound(out_true, out_false)
            else:
                raise NotImplementedError()

        loss = self.sim_weight * sim_loss + rec_loss + self.sd_weight * sd_loss

        encoder_decoder_optim.zero_grad()
        loss.backward()
        
        if summary:
            utils.log_gradients(Ec, self.train_summary_writer, global_step = self.global_itr)
            utils.log_gradients(Ep, self.train_summary_writer, global_step = self.global_itr)
            utils.log_gradients(D, self.train_summary_writer, global_step = self.global_itr)

        encoder_decoder_optim.step()

        if summary:
            self.train_summary_writer.add_images("predicted_image",x_rec[:10], global_step = self.global_itr)
            self.train_summary_writer.add_images("target_image", x_p1[:10], global_step = self.global_itr)
            self.train_summary_writer.add_scalar("sim_loss", sim_loss, global_step = self.global_itr)
            self.train_summary_writer.add_scalar("sd_loss", sd_loss, global_step = self.global_itr)
            self.train_summary_writer.add_scalar("recon_loss", rec_loss, global_step = self.global_itr)
            self.train_summary_writer.add_scalar("total_loss", loss, global_step = self.global_itr)

    def eval_step(self):
        Ec = self.nets["content_encoder"]
        Ep = self.nets["position_encoder"]
        D = self.nets["decoder"]

        with torch.autograd.no_grad():
            #Checking disentanglement
            x_source = next(self.test_generator).cuda()[:10].transpose(0,1) 
            x_target = next(self.test_generator).cuda()[:10].transpose(0,1)
            h_c = Ec(x_target[0:self.content_frames].squeeze(0))
            position_list = []
            for i in range(self.input_frames, self.input_frames+self.target_frames):
                h_p = Ep(x_source[i])
                position_list.append(h_p[0][None])
            x_source = x_source[:,0][:,None]
            generated_images = []
            for h_p in position_list:
                x_pred = D([h_c, torch.cat([h_p]*10, 0)])
                generated_images.append(x_pred)
            
            generated_images = torch.cat([x_target[self.input_frames-1]]+generated_images, dim = 3)
            generated_images = list(map(lambda x: x.squeeze(0), generated_images.split(1,0)))
            generated_images = torch.cat(generated_images, dim = 1)
            source_images = list(map(lambda x: x.squeeze(0),x_source[self.input_frames:(self.input_frames+self.target_frames),0].split(1,0)))
            source_images = torch.cat([torch.zeros_like(x_source[0,0])]+ source_images, dim = 2)
            analogy_image = torch.cat([source_images, generated_images], dim = 1)

            self.test_summary_writer.add_image("analogy_test", analogy_image, global_step = self.global_itr)
            
            #Checking reconstruction
            x = next(self.test_generator).cuda()[:10].transpose(0,1)
            k = random.randint(1,self.input_frames + self.target_frames-1)
            h_c = Ec(x[0:self.content_frames].squeeze(0))
            h_p_1 = Ep(x[1])
            h_p_2 = Ep(x[k])
            x_pred_1 = D([h_c, h_p_1])
            x_pred_2 = D([h_c, h_p_2])

            rec_image = torch.cat([x[0],x_pred_1,x_pred_2], 3)
            
            self.test_summary_writer.add_images("rec_test", rec_image, global_step = self.global_itr)

class SolverLstm(Solver):
    def __init__(self, args):
        args.deterministic = True
        encoder_checkpoint = torch.load(args.encoder_checkpoint)
        if args.dataset in ["mnist","dsprites"]:
            Ec = models.content_encoder(args.g_dims, nc = args.num_channels).cuda()
            Ep = models.pose_encoder(args.z_dims, nc = args.num_channels).cuda()
        else:
            Ec = vgg_64.encoder(args.g_dims, nc = args.num_channels).cuda()
            Ep = resnet_64.pose_encoder(args.z_dims, nc = args.num_channels).cuda()

        if args.dataset == "mpi3d_real":
            D = vgg_64.drnet_decoder(args.g_dims, args.z_dims, nc = args.num_channels).cuda()
        else:
            D = models.decoder(args.g_dims, args.z_dims, nc = args.num_channels, skips = args.skips).cuda()
        
        Ep.load_state_dict(encoder_checkpoint["position_encoder"])
        Ec.load_state_dict(encoder_checkpoint["content_encoder"])
        D.load_state_dict(encoder_checkpoint["decoder"])
        self.Ep = nn.DataParallel(Ep)
        self.Ec = nn.DataParallel(Ec)
        self.D = nn.DataParallel(D)
        self.Ep.train()
        self.Ec.train()
        self.D.train()

        lstm_model = lstm(args.g_dims + args.z_dims, args.z_dims, args.rnn_size, args.rnn_layers, args.batch_size).cuda()
        nets = {"lstm":lstm_model}

        lstm_optim = torch.optim.Adam(
            lstm_model.parameters(),
            lr = args.lr,
            betas = (args.beta1, 0.999)
        )

        optims = {"lstm_optim":lstm_optim}

        super().__init__(nets, optims, args)
    
    def train_step(self, summary = False):
        lstm_model = self.nets["lstm"]
        lstm_optim = self.optims["lstm_optim"]
        hidden = lstm_model.module.init_hidden()
        x = next(self.train_generator).cuda().transpose(0,1)

        h_c = self.Ec(x[self.input_frames-1])[0].detach()

        h_p = [self.Ep(x[i]).detach() for i in range(self.input_frames + self.target_frames)]

        mse = 0
        for i in range(1, self.input_frames + self.target_frames):
            pose_pred, hidden = lstm_model(torch.cat([h_p[i-1],h_c],1), hidden)
            mse += F.mse_loss(pose_pred,h_p[i])

        lstm_optim.zero_grad()
        mse.backward()

        if summary:
            utils.log_gradients(lstm_model, self.train_summary_writer, global_step = self.global_itr)

        lstm_optim.step()

        if summary:
            self.train_summary_writer.add_scalar("mse_loss", mse, global_step = self.global_itr)

    def eval_step(self):
        x = next(self.test_generator).cuda().transpose(0,1)
        lstm_model = self.nets["lstm"]
        with torch.autograd.no_grad():
            hidden = lstm_model.module.init_hidden()
            h_c = self.Ec(x[self.input_frames-1])
            h_p = [self.Ep(x[i]) for i in range(self.input_frames + self.target_frames-1)]
            # h_p_pred = [lstm_model(torch.cat([pose,h_c[0]],1)) for pose in h_p]
            h_p_pred = []
            for pose in h_p:
                pose_pred, hidden = lstm_model(torch.cat([pose,h_c[0]],1), hidden)
                h_p_pred.append(pose_pred)
            x_pred = [self.D([h_c,pose]) for pose in h_p_pred]
            x_pred = torch.stack(x_pred, 0)

            self.test_summary_writer.add_video("target_video", x[1:,:5].transpose(0,1), global_step = self.global_itr)
            self.test_summary_writer.add_video("predicted_video", x_pred[:,:5].transpose(0,1), global_step = self.global_itr)
