import torch
from torch import nn
import numpy as np


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=60, omega0=0.1):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.register_buffer(
            'frequencies',
            omega0 * (2.0 ** torch.arange(n_harmonic_functions)),
        )

    def forward(self, x):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        embed = (x[..., None] * self.frequencies).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)


class relu_MLP(nn.Module): # adapted from MetaFC

    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False,
                 position_embedding=False,n_harmonic_functions=60,omega0=0.5):
        super().__init__()

        self.net = []
        self.position_embedding = position_embedding

        if position_embedding == False:
            self.net.append(nn.Linear(in_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))
        else:
            self.net.append(HarmonicEmbedding(n_harmonic_functions=n_harmonic_functions,omega0=omega0))
            self.net.append(nn.Linear(in_features*n_harmonic_functions*2, hidden_features))
            self.net.append(nn.ReLU(inplace=True))


        for i in range(num_hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(nn.ReLU(inplace=True))

        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        else:
            self.net.append(nn.Linear(hidden_features, out_features))
            self.net.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, coords):
        '''Simple forward pass without computation of spatial gradients.'''
        output = self.net(coords)
        return output,coords



if __name__ == '__main__':
    model = relu_MLP(in_features=2, out_features=1,
                 num_hidden_layers=3, hidden_features=256,
                 outermost_linear=True, position_embedding=True)
    print(model)