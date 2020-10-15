import torch
from torch import nn

import models
import lstm
import utils

class TimeVariantLatentCritic(nn.Module):

    def __init__(self, z_dim, nout = 1):
        super().__init__()
        self.z_dim = z_dim
        self.nout = nout
        self.net_tcn = nn.Sequential(
            nn.Conv1d(z_dim, 128, 2, stride = 2), ##Input 10x16
            nn.LeakyReLU(0.2,True),
            nn.Conv1d(128, 256, 2, stride = 2), #Input 128X8
            nn.LeakyReLU(0.2,True),
            nn.Conv1d(256, 512, 2, stride = 2), #Input 256X4
            nn.LeakyReLU(0.2,True),
            nn.Conv1d(512, 1024, 2, stride = 2), #Input 512X2
            nn.LeakyReLU(0.2,True)            
        )

        self.net_fc = nn.Sequential(
            nn.Linear(1024, 64),
            nn.LeakyReLU(0.2,True),
            nn.Linear(64, nout)
        )

        def init_weights(m):
            classname = m.__class__.__name__

            if classname.find("Linear") != -1 or classname.find("Conv") != -1:
                nn.init.xavier_normal_(m.weight, gain = nn.init.calculate_gain("leaky_relu",0.2))
        
        self.net_tcn.apply(init_weights)
        self.net_fc.apply(init_weights)

    @property
    def single_negative_sample(self):
        return True


    def forward(self, joint_samples):
        def apply_critic(z):
            z = z.transpose(1,2)
            hidden = self.net_tcn(z).squeeze(2)
            return self.net_fc(hidden).squeeze()

        marginal_samples = utils.shuffle_time(joint_samples)

        f_true = apply_critic(joint_samples)
        f_false = apply_critic(marginal_samples)

        return f_true, f_false

class ImageLatentCritic(nn.Module):
    
    def __init__(self, z_dim, frame_encoding = 128,nc = 1, nout = 1):
        super().__init__()
        self.frame_encoder = models.pose_encoder(frame_encoding, nc = nc)
        self.z_embed_net = nn.Linear(z_dim, 128)

        self.net_fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, nout)
        )

    def forward(self, joint_samples):
        def apply_critic(images, z):
            image_hidden = self.frame_encoder(images)
            z_hidden = self.z_embed_net(z)
            output = self.net_fc(torch.cat([image_hidden,z_hidden],1)).squeeze()
            return output
        
        assert(len(joint_samples) == 2)

        marginal_samples = utils.shuffle_dim(joint_samples)

        f_true = apply_critic(*joint_samples)
        f_false = apply_critic(*marginal_samples)

        return f_true, f_false