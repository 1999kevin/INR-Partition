import torch.nn as nn
from models.siren_metabase import Siren
import torch.nn.functional as F
import torch
from torchmeta.modules import (MetaModule)


class meta_multi_MLP(MetaModule):  # adapted from MetaFC

    def __init__(self, in_features, out_features, n_heads=10, segment_weight=None, hidden_features=256, hidden_layers=3,
                 omega=30):
        super().__init__()
        self.n_heads = n_heads
        self.heads = []
        for i in range(n_heads):
            self.heads.append(Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features,
                                    hidden_layers=hidden_layers, outermost_linear=True, first_omega_0=omega,
                                    hidden_omega_0=omega))
        self.layer_module = nn.ModuleList(self.heads)
        self.segment_weight = F.one_hot(segment_weight).cuda()
        self.out_features = out_features

    def update_mask(self, segment_weight=None,
                    average=False):  # batch_size*img_flat_size*1 -> batch_size*img_flat_size*1*n_subdomain
        if average == False:
            self.segment_weight = F.one_hot(segment_weight).cuda()
            # self.segment_weight = segment_weight.cuda()
        else:
            b, l, d = segment_weight.shape
            self.segment_weight = torch.ones((b, l, d, self.n_heads)) / self.n_heads

            self.segment_weight = self.segment_weight.cuda()


    def forward(self, coords, params=None):
        '''Simple forward pass without computation of spatial gradients.'''

        output = None
        for i, layer in enumerate(self.layer_module):

            if output == None:
                output = self.segment_weight[:, :, :, i] * layer(coords,
                                                                 params=self.get_subdict(params, f'layer_module.{i}'))
            else:
                output += self.segment_weight[:, :, :, i] * layer(coords,
                                                                  params=self.get_subdict(params, f'layer_module.{i}'))
        return output

        # output_final = torch.zeros(coords.shape[0], coords.shape[1], self.out_features).cuda()
        # for b in range(coords.shape[0]):
        #     for i in range(self.n_heads):
        #         index = torch.where(self.segment_weight[b, :, 0] == i)[0]
        #
        #         output = self.layer_module[i](coords[b, index, :], params=self.get_subdict(params, f'layer_module.{i}'))
        #         output_final[b, index, :] = output[b, :, :]

        # return output_final
