import numpy as np
from os.path import join as pjoin
import os
import matplotlib.pyplot as plt


result_dirs = './001_L_save_results/'

siren_name_list = ['Siren','multi-Sirenhfs9','multi-Siren9interval']
label_name_list = ['SIREN','SIREN with PoS#9','SIREN with PoG#9']

plt.figure(figsize=(7, 5.5))

for i,name in enumerate(siren_name_list):
    psnr = np.load(result_dirs+name+'_psnr.npy')
    plt.plot(psnr, label=label_name_list[i])


plt.xlabel('step',fontsize=16)
plt.ylabel('PSNR',fontsize=16)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=16,loc=4)

# plt.savefig('./001_L_save_results/001_L_siren.png')
plt.show()


relu_name_list = ['relu-mlp','multi-reluhfs9','multi-relu9interval']
label_name_list = ['ReLU-MLP','ReLU-MLP with PoS#9','ReLU-MLP with PoG#9']

plt.figure(figsize=(7, 5.5))
for i,name in enumerate(relu_name_list):
    psnr = np.load(result_dirs+name+'_psnr.npy')
    plt.plot(psnr, label=label_name_list[i])


plt.xlabel('step',fontsize=16)
plt.ylabel('PSNR',fontsize=16)
plt.legend(fontsize=16,loc=4)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# plt.savefig('./001_L_save_results/001_L_relu.png')
plt.show()