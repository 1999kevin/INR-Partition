import sys
sys.path.append('../')

import matplotlib.pyplot as plt 
from skimage import io
import cv2
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.relu_mlp import relu_MLP
from models.siren import Siren
from models.multi_mlp import multi_MLP
import torch.nn.functional as F

from  masks.hfs_segment_plus import hfs_domain_decompostion
from masks.inteval_segment import grid_mask
from utils.metrics import calc_psnr, calculate_ssim
import time
import argparse



def lin2img(tensor, width, height):
    batch_size, num_samples, channels = tensor.shape
    return tensor.view(batch_size, height, width, channels).squeeze(-1)


def plot_sample_image(img_batch, ax,width,height):
    img = lin2img(img_batch,width,height)[0].detach().cpu().numpy()
    img = np.clip(img, 0., 1.)
    ax.set_axis_off()
    ax.imshow(img)


def get_mgrid(height, width, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    x = torch.linspace(-1, 1, steps=width)
    y = torch.linspace(-1, 1, steps=height)
    mgrid = torch.stack(torch.meshgrid(x, y), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class ImageFitting_color(Dataset):
    def __init__(self, img):
        super().__init__()
        self.pixels = img.view(-1, 3)
        self.coords = get_mgrid(height=img.shape[0], width=img.shape[1], dim=2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


def train(model,total_steps,steps_til_summary, dataloader,height=120,width=160,lr=1e-4, writer=None,log_save_path=None):
    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    model_input.requires_grad_(True)
    loss_list = []
    psnr_list = []
    time_list = []

    start_time = time.time()


    for step in range(total_steps):
        model_output, coords = model(model_input)
        loss = ((model_output - ground_truth)**2).mean()

        psnr = calc_psnr(model_output, ground_truth)

        loss_list.append(loss.item())
        psnr_list.append(psnr.item())

        # print(step)
        if writer != None:
            writer.add_scalar('loss', loss, step)
            writer.add_scalar('psnr', psnr, step)
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            # psnr = calc_psnr(model_output, ground_truth)
            # ssim = calculate_ssim(model_output, ground_truth)

            fig, axes = plt.subplots(1, 1, figsize=(6, 6))
            plot_sample_image(model_output, ax=axes, width=width, height=height)

            plt.show()



        optim.zero_grad()
        loss.backward()
        optim.step()

        end_time = time.time()
        run_time = end_time-start_time
        time_list.append(run_time)

    loss_array = np.array(loss_list)
    psnr_array = np.array(psnr_list)
    time_array = np.array(time_list)

    np.save(log_save_path + '_loss.npy', loss_array)
    np.save(log_save_path + '_psnr.npy', psnr_array)
    np.save(log_save_path + '_time.npy', time_array)






if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='single_image_fit')

    # model config
    parser.add_argument('--architecture', type=str, required=False, default='Siren', help='architecture, options:[relu-mlp, Siren, multi-relu, multi-Siren]')
    parser.add_argument('--mask', type=str, required=False, default=None, help='mask, options:[4interval, 9interval, 16interval, hfs4, hfs9]')
    parser.add_argument('--hidden_features', type=int, required=False, default=512, help='hidden features of one head')
    parser.add_argument('--hidden_layers', type=int, required=False, default=3, help='hidden layers of one head')

    # data config 
    parser.add_argument('--img_path', type=str, required=False, default='../data/001_L.png', help='path to the image')
    parser.add_argument('--log_save_path', type=str, required=False, default='./001_L_save_results/', help='path to save the results')
    parser.add_argument('--epoch', type=int, required=False, default=1500,
                        help='the training epoch')
    parser.add_argument('--steps_til_summary', type=int, required=False, default=100,
                        help='step for visualization')

    args = parser.parse_args()


    mask = args.mask
    architecture = args.architecture
    hidden_features = args.hidden_features
    hidden_layers = args.hidden_layers
    img_path = args.img_path
    log_save_path = args.log_save_path


    lr = 1e-4

    img1 = io.imread(img_path)
    img1 = np.array(img1)/255
    height = img1.shape[0]
    width = img1.shape[1]
    plt.imshow(img1)
    plt.show()

    if mask == 'hfs4':
        segment1 = hfs_domain_decompostion(cv2.imread(img_path), patch_num=4).astype(np.int_)
    elif mask == 'hfs9':
        segment1 = hfs_domain_decompostion(cv2.imread(img_path),patch_num=9).astype(np.int_)

    elif mask == '4interval':
        segment1, _ = grid_mask(height, width, grid_num=2)
    elif mask == '9interval':
        segment1,_ = grid_mask(height,width,grid_num=3)
    elif mask == '16interval':
        segment1,_ = grid_mask(height,width,grid_num=4)
    elif mask == None:
        segment1 = None

    if segment1 is not None:
        plt.imshow(segment1)
        plt.show()
        segment1 = torch.Tensor(segment1)
        segment1 = segment1.long()
        n_heads = torch.max(segment1) + 1
        print(n_heads)



    dataset_img = ImageFitting_color(torch.Tensor(img1))
    dataloader = DataLoader(dataset_img, batch_size=1, pin_memory=True, num_workers=0)



    if architecture == 'multi-Siren':
        model = multi_MLP(n_heads=n_heads, type='siren', segment_weight=segment1,in_features=2, out_features=3,
                          hidden_features=hidden_features, hidden_layers=hidden_layers)
    elif architecture == 'multi-relu':
        model = multi_MLP(n_heads=n_heads, type='relu_mlp', segment_weight=segment1, in_features=2, out_features=3,
                          hidden_features=hidden_features,hidden_layers=hidden_layers)

    elif architecture == 'Siren':
        model = Siren(in_features=2, out_features=3, hidden_features=hidden_features, hidden_layers=hidden_layers,
              outermost_linear=True, first_omega_0=60, hidden_omega_0=30)
        mask = ''
    elif architecture =='relu-mlp':
        model = relu_MLP(in_features=2, out_features=3, num_hidden_layers=hidden_layers,
                                           hidden_features=hidden_features, outermost_linear=True, position_embedding=True)
        mask = ''



    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    model.cuda()
    total_steps = args.epoch
    steps_til_summary = args.steps_til_summary


    log_save_path = log_save_path + architecture + mask

    train(model, total_steps, steps_til_summary, dataloader, height=height,width=width, lr=lr,log_save_path=log_save_path)
