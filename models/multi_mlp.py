import sys
sys.path.append('../')

import torch.nn as nn
from models.siren import Siren
from models.relu_mlp import relu_MLP
import torch.nn.functional as F
from masks.inteval_segment import grid_mask
import torch
import time




class multi_MLP(nn.Module):  # adapted from MetaFC

    def __init__(self, n_heads=10, type='relu_mlp', segment_weight=None,in_features=2, out_features=3,hidden_features=256, hidden_layers=3):
        super().__init__()
        self.n_heads = n_heads
        self.heads = []
        self.out_features = out_features
        for i in range(n_heads):
            if type == 'siren':
                self.heads.append(Siren(in_features=in_features, out_features=out_features, hidden_features=hidden_features, hidden_layers=hidden_layers,
                                        outermost_linear=True, first_omega_0=60, hidden_omega_0=30))
            elif type == 'relu_mlp':
                self.heads.append(relu_MLP(in_features=2, out_features=3, num_hidden_layers=hidden_layers,
                                           hidden_features=hidden_features, outermost_linear=True, position_embedding=True))
            else:
                print('no such type')

        self.layer_module = nn.ModuleList(self.heads)
        self.segment_weight = segment_weight.view(1, -1).cuda()
        print('shape:',F.one_hot(segment_weight.view(1,-1)).shape)

    def forward(self, coords):
        '''Simple forward pass without computation of spatial gradients.'''
        heads_index = []
        coords_list = []

        output_list = []
        output_final = torch.zeros(1, coords.shape[1],self.out_features).cuda()
        for i in range(self.n_heads):
            index = torch.where(self.segment_weight[0,:]==i)[0]
            heads_index.append(index)

            coords_list.append(coords[:,index,:])

            output,_ = self.layer_module[i](coords[:,index,:])
            output_list.append(output)

            output_final[0,index,:] = output[0,:,:]

        return output_final,coords



if __name__ == '__main__':
    architecture = 'multi-MLP'
    n_heads = 1
    segment1, _ = grid_mask(216, 216, grid_num=2)
    segment1 = torch.Tensor(segment1)
    segment1 = segment1.long()
    hidden_features = 512
    hidden_layers = 8

    if architecture == 'multi-MLP':
        model = multi_MLP(n_heads=n_heads, type='siren', segment_weight=segment1,hidden_features=hidden_features, hidden_layers=hidden_layers).cuda()
    elif architecture == 'multi-relu':
        model = multi_MLP(n_heads=n_heads, type='relu_mlp', segment_weight=segment1, hidden_features=hidden_features,
                            hidden_layers=hidden_layers).cuda()



    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)


