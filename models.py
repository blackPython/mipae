import torch
import torch.nn as nn

from lstm import lstm, gaussian_lstm

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)


class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class pose_encoder(nn.Module):
    def __init__(self, pose_dim, nc=1, normalize=False):
        super(pose_encoder, self).__init__()
        nf = 64
        self.normalize = normalize
        self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                dcgan_conv(nc, nf),
                # state size. (nf) x 32 x 32
                dcgan_conv(nf, nf * 2),
                # state size. (nf*2) x 16 x 16
                dcgan_conv(nf * 2, nf * 4),
                # state size. (nf*4) x 8 x 8
                dcgan_conv(nf * 4, nf * 8),
                # state size. (nf*8) x 4 x 4
                nn.Conv2d(nf * 8, pose_dim, 4, 1, 0),
                nn.BatchNorm2d(pose_dim),
                nn.Tanh()
                )

    def forward(self, input):
        output = self.main(input).squeeze()
        if self.normalize:
            return nn.functional.normalize(output, p=2)
        else:
            return output

class content_encoder(nn.Module):
    def __init__(self, content_dim, nc=1):
        super(content_encoder, self).__init__()
        nf = 64
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 32 x 32
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 4 x 4
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, content_dim, 4, 1, 0),
                nn.BatchNorm2d(content_dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.squeeze(), [h1, h2, h3, h4]


class decoder(nn.Module):
    def __init__(self, content_dim, pose_dim, nc=1, skips = True):
        super(decoder, self).__init__()
        nf = 64
        self.skips = skips
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(content_dim+pose_dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        mul_coff = 2 if skips else 1

        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * mul_coff, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 * mul_coff, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 * mul_coff, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
        nn.ConvTranspose2d(nf * mul_coff, nc, 4, 2, 1),
        nn.Sigmoid()
        # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        content, pose = input
        content, skip = content
        d1 = self.upc1(torch.cat([content, pose], 1)[:,:,None,None])
        if self.skips:
            d2 = self.upc2(torch.cat([d1, skip[3]], 1))
            d3 = self.upc3(torch.cat([d2, skip[2]], 1))
            d4 = self.upc4(torch.cat([d3, skip[1]], 1))
            output = self.upc5(torch.cat([d4, skip[0]], 1))
        else:
            d2 = self.upc2(d1)
            d3 = self.upc3(d2)
            d4 = self.upc4(d3)
            output = self.upc5(d4)
        return output

class lp_decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super().__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        # state size. (nf*8) x 4 x 4
        self.upc2 = dcgan_upconv(nf * 8 * 2, nf * 4)
        # state size. (nf*4) x 8 x 8
        self.upc3 = dcgan_upconv(nf * 4 * 2, nf * 2)
        # state size. (nf*2) x 16 x 16
        self.upc4 = dcgan_upconv(nf * 2 * 2, nf)
        # state size. (nf) x 32 x 32
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nc, 4, 2, 1),
                nn.Sigmoid()
                # state size. (nc) x 64 x 64
                )

    def forward(self, input):
        vec, skip = input
        d1 = self.upc1(vec.view(-1, self.dim, 1, 1))
        d2 = self.upc2(torch.cat([d1, skip[3]], 1))
        d3 = self.upc3(torch.cat([d2, skip[2]], 1))
        d4 = self.upc4(torch.cat([d3, skip[1]], 1))
        output = self.upc5(torch.cat([d4, skip[0]], 1))
        return output


class scene_discriminator(nn.Module):
    def __init__(self, pose_dim, nf=512):
        super(scene_discriminator, self).__init__()
        self.pose_dim = pose_dim
        self.main = nn.Sequential(
                nn.Linear(pose_dim*2, nf),
                nn.ReLU(True),
                nn.Linear(nf, nf),
                nn.ReLU(True),
                nn.Linear(nf, 1),
                )

    def forward(self, input):
        output = self.main(torch.cat(input, 1).view(-1, self.pose_dim*2))
        return output

class vae_scene_discriminator(nn.Module):
    def __init__(self, pose_dim, nf = 1024):
        super().__init__()
        self.pose_dim = pose_dim
        self.main = nn.Sequential(
            nn.Linear(pose_dim*2, nf),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, nf),
            nn.LeakyReLU(0.2, True),
            nn.BatchNorm1d(nf),
            nn.Linear(nf, 1),
        )
    
    def forward(self, input):
        output = self.main(torch.cat(input, 1).view(-1, self.pose_dim*2))
        return output

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes, nf = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, nf),
            nn.ReLU(True),
            nn.Linear(nf, num_classes)
        )

    def forward(self,x):
        return self.net(x)

class content_encoder_lstm(nn.Module):
    def __init__(self, g_dims, content_encoder, batch_size, rnn_size = 256, rnn_layers = 1):
        super().__init__()
        self.content_encoder = content_encoder
        self.content_rnn = lstm(g_dims, g_dims, rnn_size, rnn_layers, batch_size)

    def forward(self, x):
        x.transpose_(0,1)
        B = x.shape[1]
        lstm_hidden = self.content_rnn.init_hidden(batch_size=B)
        for i in range(x.shape[0]):
            hidden, skip = self.content_encoder(x[i])
            h_content, lstm_hidden = self.content_rnn(hidden, lstm_hidden)
        
        return h_content, skip