import torch
import sys
sys.path.append('../')
from models.siren import Siren


from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import math


def plot_scatter(x, y):
    plt.scatter(x, y, s=1)
    plt.show()

def plot_scatter2D(y,size=256):
    y2D = y.reshape(size,size)
    y2D = y2D/2+0.5
    plt.imshow(y2D)
    plt.show()


def compare_loss(losses_list, multi_losses_list, index=0):
    plt.plot(losses_list[index], label='siren')
    plt.plot(multi_losses_list[index], label='multi')
    plt.legend()
    plt.show()


def boundaries(x1,x2):
    return 2*x1*x2+x1+x2

def step(x, omega):
    return np.floor(x*omega)/omega

def boundary_function(x, omega_array=None):
    y_count = [np.sum(np.where(x[i] < omega_array, 1, 0)) for i in range(x.shape[0])]
    y = np.where(np.array(y_count) % 2==0, 1, -1)
    return y


class random_step_dataset2D(Dataset):
    def __init__(self, bound=[-1, 1], offset=0, size=1000, nheads=4,omega_array1=None,omega_array2=None):
        super().__init__()

        self.bound = bound
        # self.omega = omega
        self.offset = offset
        self.size = size
        self.nheads = nheads
        self.omega_array1 = omega_array1
        self.omega_array2 = omega_array2

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        x1 = torch.linspace(-1, 1, steps=self.size)
        x2 = torch.linspace(-1, 1, steps=self.size)
        mgrid = torch.stack(torch.meshgrid(x1, x2), dim=-1)
        x = mgrid.reshape(-1, 2)

        x1 = x1.numpy()
        x2 = x2.numpy()

        y1 = boundary_function(x1,self.omega_array1)
        y2 = boundary_function(x2, self.omega_array2)
        y1 = np.expand_dims(y1, axis=0)
        y2 = np.expand_dims(y2, axis=1)

        y = y1 * y2
        y = torch.Tensor(y).reshape(-1, 1)
        # define masks
        masks = np.zeros(x.shape[0])

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float), torch.tensor(masks, dtype=torch.long)


def square_root_as_int(x):
  return int(math.sqrt(x))

def factors_of_(number):
  factors = []
  for x in range(square_root_as_int(number) + 1,0,-1):
    if number % x == 0:
      return x, number/x


def train_siren_step(omega, max_step=3000, threhold=1e-3,type='step'):
    size = 256
    if type == 'random_step2D':
        omega1,omega2 = factors_of_(omega)
        omega_array1 = np.random.uniform(-1, 1, int(omega1))
        omega_array2 = np.random.uniform(-1, 1, int(omega2))

        sine_ds = random_step_dataset2D(bound=[-1, 1], offset=0, size=size, nheads=4,omega_array1=omega_array1,omega_array2=omega_array2)
        sine_ds_test = random_step_dataset2D(bound=[-1, 1], offset=0, size=size, nheads=4,omega_array1=omega_array1,omega_array2=omega_array2)

    net = Siren(in_features=2, out_features=1, hidden_features=32, hidden_layers=3,
                outermost_linear=True, first_omega_0=30, hidden_omega_0=30)

    net = net.cuda()

    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    Loss = nn.MSELoss()
    print_step = 1

    loss_list = []
    epoch = 0

    x, y, _ = sine_ds_test[0]
    # plot_scatter2D(y,size)

    while epoch < max_step:
        loss = None
        x, y, _ = sine_ds[0]
        x, y = x.cuda(), y.cuda()

        y_predict, _ = net(x)
        loss = Loss(y_predict, y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())

        if (epoch) % print_step == 0:
            x, y, _ = sine_ds_test[0]
            x, y = x.cuda(), y.cuda()
            y_predict, _ = net(x)
            loss = Loss(y_predict, y)
            if loss < threhold:
                break
        epoch += 1


    print(omega, epoch)
    return epoch



if __name__ == '__main__':
    step_list_total = []

    x_origin = range(10, 260, 10)
    bounds = []
    x = []
    for i in range(len(x_origin)):
        x1, x2 = factors_of_(x_origin[i])

        b = boundaries(x1, x2)
        if b not in bounds:
            bounds.append(b)
            x.append(x_origin[i])
        else:
            print(i, x_origin[i])

    for i in range(5):
        print('i:', i)
        step_list = []
        for omega in x:  # random_step
            print('omega:', omega)
            step_list.append(train_siren_step(omega,max_step=4000, threhold=0.05,type='random_step2D'))
        step_list_total.append(step_list)

    a = np.array(step_list_total)
    np.save('random_step2D_new.npy', a)


