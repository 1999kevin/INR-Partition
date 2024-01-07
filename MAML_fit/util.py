import numpy as np
import torch
from collections import OrderedDict, Mapping


def calc_psnr(img1, img2):
    if isinstance(img1, torch.Tensor):
        return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))
    else:
        return 10. * np.log10(1. / np.mean((img1 - img2) ** 2))



def sample_repeat(sample, times=4):
    for key in sample.keys():
        for key2 in sample[key].keys():
            sample[key][key2] = sample[key][key2].repeat(times, 1, 1)
    return sample


def get_mgrid(sidelen):
    # Generate 2D pixel coordinates from an image of sidelen x sidelen
    pixel_coords = np.stack(np.mgrid[:sidelen, :sidelen], axis=-1)[None, ...].astype(np.float32)
    pixel_coords /= sidelen
    pixel_coords -= 0.5
    pixel_coords = torch.Tensor(pixel_coords).view(-1, 2)
    return pixel_coords


def l2_loss(prediction, gt):
    return ((prediction - gt) ** 2).mean()

def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.view(batch_size, sidelen, sidelen, channels).squeeze(-1)


def plot_sample_image(img_batch, ax):
    img = lin2img(img_batch)[0].detach().cpu().numpy()
    img += 1
    img /= 2.
    img = np.clip(img, 0., 1.)
    ax.set_axis_off()
    ax.imshow(img)


def dict_to_gpu(ob):
    if isinstance(ob, Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    else:
        return ob.cuda()


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
