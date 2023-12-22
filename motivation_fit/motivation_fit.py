import torch
import sys
sys.path.append('../')

from models.siren import Siren
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn



def plot_scatter(x, y):
    plt.scatter(x, y, s=1)
    plt.show()


def compare_loss(losses_list, multi_losses_list, index=0):
    plt.plot(losses_list[index], label='siren')
    plt.plot(multi_losses_list[index], label='multi')
    plt.legend()
    plt.show()




def boundary_function(x, omega_array=None):
    y_count = [np.sum(np.where(x[i] < omega_array, 1, 0)) for i in range(x.shape[0])]
    y = np.where(np.array(y_count) % 2==0, 1, -1)

    return y




class random_step_dataset(Dataset):
    def __init__(self, bound=[-1, 1], omega=10, offset=0, size=1000, nheads=4, split='train',omega_array=None):
        super().__init__()

        self.bound = bound
        self.omega = omega
        self.offset = offset
        self.size = size
        self.nheads = nheads
        self.split = split
        self.omega_array = omega_array


    def __len__(self):
        return 1


    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        if self.split == 'train':
            x = np.random.uniform(low=self.bound[0], high=self.bound[1], size=self.size)
        elif self.split == 'test':
            step = (self.bound[1] - self.bound[0]) / self.size
            x = np.mgrid[self.bound[0]:self.bound[1]:step]
        x = np.expand_dims(x, axis=1)


        y = boundary_function(x,self.omega_array)
        y = np.expand_dims(y, axis=1)
        # define masks
        masks = np.zeros(x.shape[0])
        width = (self.bound[1] - self.bound[0]) / self.nheads
        for i in range(self.nheads):
            for j in range(x.shape[0]):
                if x[j] > self.bound[0] + i * width and x[j] < self.bound[0] + (i + 1) * width:
                    masks[j] = i

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float), torch.tensor(masks, dtype=torch.long)




def train_siren_step(omega, max_step=3000, threhold=1e-3,type='step'):
    if type == 'random_step':
        omega_array = np.random.uniform(-1, 1, omega)
        sine_ds = random_step_dataset(bound=[-1, 1], omega=omega, offset=0, size=5000, nheads=4,split='train', omega_array=omega_array)
        sine_ds_test = random_step_dataset(bound=[-1, 1], omega=omega, offset=0, size=5000, nheads=4, split='test',omega_array=omega_array)

    net = Siren(in_features=1, out_features=1, hidden_features=32, hidden_layers=3,
                outermost_linear=True, first_omega_0=10, hidden_omega_0=30)
    net = net.cuda()

    optim = torch.optim.Adam(net.parameters(), lr=0.001)
    Loss = nn.MSELoss()
    print_step = 1

    loss_list = []
    epoch = 0

    # x, y, _ = sine_ds[0]
    # plot_scatter(x, y)

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
    for i in range(5):
        print('i:', i)
        step_list = []
        for omega in range(1, 61, 2):  # random_step
            print('omega:', omega)
            step_list.append(train_siren_step(omega,max_step=4000, threhold=0.05,type='random_step'))
        step_list_total.append(step_list)

    a = np.array(step_list_total)
    np.save('random_step.npy', a)


