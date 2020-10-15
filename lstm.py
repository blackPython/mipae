import torch
import torch.nn as nn
from torch.autograd import Variable

class lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size, normalize=False):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.output = nn.Sequential(
                nn.Linear(hidden_size, output_size),
                #nn.BatchNorm1d(output_size),
                nn.Tanh())
        self.normalize = normalize

    def init_hidden(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))
        return hidden

    def forward(self, input, hidden):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]

        output = self.output(h_in)
        if self.normalize:
            return nn.functional.normalize(output, p=2), hidden
        else:
            return output, hidden

class gaussian_lstm(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, batch_size):
        super(gaussian_lstm, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers
        self.embed = nn.Linear(input_size, hidden_size)
        self.lstm = nn.ModuleList([nn.LSTMCell(hidden_size,hidden_size) for i in range(self.n_layers)])
        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)

    def init_hidden(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size
        hidden = []
        for i in range(self.n_layers):
            hidden.append((Variable(torch.zeros(batch_size, self.hidden_size).cuda()),
                           Variable(torch.zeros(batch_size, self.hidden_size).cuda())))

        return hidden

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.normal(torch.zeros_like(std), torch.ones_like(std))
        return eps.mul(std).add_(mu)
    
    def forward(self, input, hidden):
        embedded = self.embed(input.view(-1, self.input_size))
        h_in = embedded
        for i in range(self.n_layers):
            hidden[i] = self.lstm[i](h_in, hidden[i])
            h_in = hidden[i][0]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in).clamp(-5,5)
        z = self.reparameterize(mu, logvar)
        std = logvar.mul(0.5).exp_()
        return z, mu, std, hidden