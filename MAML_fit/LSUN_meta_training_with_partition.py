import sys
sys.path.append('../')

import torch.nn as nn
import torchvision
from models.multi_mlp_metabase import meta_multi_MLP
from masks.hfs_segment_plus import hfs_domain_decompostion
from masks.inteval_segment import grid_mask

from torchvision import transforms
from util import *

class LSUN_mask():
    def __init__(self, size=256, root='./', classes='church_outdoor_train',
                 n_subdomain=4, mask_method='grid'):
        self.transform = transforms.Compose([
            transforms.CenterCrop(size),
        ])

        self.dataset = torchvision.datasets.LSUN(root=root, classes=[classes],
                                                 transform=self.transform)
        self.meshgrid = get_mgrid(sidelen=size)
        self.n_subdomain = n_subdomain
        self.mask_method = mask_method
        self.size = size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        img, _ = self.dataset[item]
        img_numpy = np.array(img)
        if self.mask_method == 'hfs':
            mask = hfs_domain_decompostion(img_numpy, patch_num=self.n_subdomain)
        elif self.mask_method == 'grid':
            mask, _ = grid_mask(self.size,self.size, grid_num=2)
        elif self.mask_method == 'None':
            mask = None
        else:
            print('no implement')
        img = transforms.ToTensor()(img_numpy)

        img_flat = img.permute(1, 2, 0).view(-1, 3)
        if self.mask_method != 'None':
            mask_flat = torch.Tensor(mask.astype(np.int32)).view(-1, 1).long()
            return {'context': {'x': self.meshgrid, 'y': img_flat, 'mask': mask_flat},
                    'query': {'x': self.meshgrid, 'y': img_flat, 'mask': mask_flat}}
        else:
            return {'context': {'x': self.meshgrid, 'y': img_flat},
                    'query': {'x': self.meshgrid, 'y': img_flat}}


class MAML_multi_partition(nn.Module):
    def __init__(self, num_meta_steps, hypo_module, loss, init_lr,
                 lr_type='static', first_order=False):
        super().__init__()

        self.hypo_module = hypo_module  # The module who's weights we want to meta-learn.

        self.first_order = first_order
        self.loss = loss
        self.lr_type = lr_type
        self.log = []

        self.register_buffer('num_meta_steps', torch.Tensor([num_meta_steps]).int())

        if self.lr_type == 'static':
            self.register_buffer('lr', torch.Tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.Tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.Tensor([init_lr]))
                                        for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter':  # As proposed in "Meta-SGD".
            self.lr = nn.ParameterList([])
            hypo_parameters = hypo_module.parameters()
            for param in hypo_parameters:
                self.lr.append(nn.Parameter(torch.ones(param.size()) * init_lr))
        elif self.lr_type == 'per_parameter_per_step':
            multi_hypo_module = meta_multi_MLP(in_features=2, hidden_features=128, hidden_layers=3,
                                                 out_features=3, n_heads=4,
                                                 segment_weight=torch.ones((256, 256), dtype=int)).cuda()

            self.lr = nn.ModuleList([])
            for name, param in multi_hypo_module.meta_named_parameters():
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))

        elif self.lr_type == 'per_parameter_per_step_single_head':
            self.lr = nn.ModuleList([])
            self.lr_dict = {}
            count = 0
            for name, param in self.hypo_module.meta_named_parameters():
                print(name)
                self.lr_dict[name] = count
                count += 1
                self.lr.append(nn.ParameterList([nn.Parameter(torch.ones(param.size()) * init_lr)
                                                 for _ in range(num_meta_steps)]))

        param_count = 0
        for param in self.parameters():
            param_count += np.prod(param.shape)

        print(param_count)

    def _update_step(self, loss, param_dict, step):
        grads = torch.autograd.grad(loss, param_dict.values(),
                                    create_graph=False if self.first_order else True)
        params = OrderedDict()
        for i, ((name, param), grad) in enumerate(zip(param_dict.items(), grads)):
            if self.lr_type in ['static', 'global']:
                lr = self.lr
                params[name] = param - lr * grad
            elif self.lr_type in ['per_step']:
                lr = self.lr[step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter']:
                lr = self.lr[i]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step']:
                lr = self.lr[i][step]
                params[name] = param - lr * grad
            elif self.lr_type in ['per_parameter_per_step_single_head']:
                name_sub = name[15:]
                lr = self.lr[self.lr_dict[name_sub]][step]
                params[name] = param - lr * grad
            else:
                raise NotImplementedError

        return params, grads

    #     def forward_with_params(self, query_x, fast_params, **kwargs):
    #         output = self.hypo_module(query_x, params=fast_params)
    #         return output

    def generate_params(self, context_dict):
        """Specializes the model"""
        x = context_dict.get('x').cuda()
        y = context_dict.get('y').cuda()

        meta_batch_size = x.shape[0]

        multi_hypo_module = meta_multi_MLP(in_features=2, hidden_features=128, hidden_layers=3,
                                             out_features=3, n_heads=4,
                                             segment_weight=torch.ones((256, 256), dtype=int)).cuda()

        with torch.enable_grad():
            # First, replicate the initialization for each batch item.
            # This is the learned initialization, i.e., in the outer loop,
            # the gradients are backpropagated all the way into the
            # "meta_named_parameters" of the hypo_module.
            fast_params = OrderedDict()
            for name, param in self.hypo_module.meta_named_parameters():
                fast_params[name] = param[None, ...].repeat((meta_batch_size,) + (1,) * len(param.shape))

            # Here update multi-head module and copy each siren to multi-head
            multi_hypo_module.update_mask(context_dict.get('mask').cuda())

            multi_fast_params = OrderedDict()
            for name, param in multi_hypo_module.meta_named_parameters():
                multi_fast_params[name] = fast_params[name[15:]].clone()

            prev_loss = 1e6
            intermed_predictions = []

            for j in range(self.num_meta_steps):
                # Using the current set of parameters, perform a forward pass with the context inputs.
                predictions = multi_hypo_module(x, params=multi_fast_params)
                # Compute the loss on the context labels.
                loss = self.loss(predictions, y)
                intermed_predictions.append(predictions.item())

                if loss > prev_loss:
                    print('inner lr too high?')

                # Using the computed loss, update the fast parameters.
                multi_fast_params, grads = self._update_step(loss, multi_fast_params, j)
                prev_loss = loss

        return multi_fast_params, intermed_predictions, multi_hypo_module

        # return multi_fast_params, multi_hypo_module

    def forward(self, meta_batch, **kwargs):
        # The meta_batch conists of the "context" set (the observations we're conditioning on)
        # and the "query" inputs (the points where we want to evaluate the specialized model)
        context = meta_batch['context']
        query_x = meta_batch['query']['x'].cuda()
        mask = context.get('mask').cuda()

        # Specialize the model with the "generate_params" function.
        multi_fast_params, intermed_predictions, multi_hypo_module = self.generate_params(context)
        # multi_fast_params, multi_hypo_module = self.generate_params(context)



        # Compute the final outputs.
        model_output = multi_hypo_module(query_x, params=multi_fast_params)
        out_dict = {'model_out': model_output, 'intermed_predictions': intermed_predictions}
        # out_dict = {'model_out': model_output}

        return out_dict

