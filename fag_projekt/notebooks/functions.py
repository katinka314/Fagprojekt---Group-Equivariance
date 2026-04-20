
import torch 
import torch.nn as nn
import torch.nn.functional  as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np
import torch.nn as nn
import kagglehub
import pandas as pd
import PIL.Image as Image
from torchvision import transforms


""" TODO: Vi skal skrive den med underfunktioner for læsbarhed """
def kernel(kernel_size: int, l: int, plot: bool = False) ->  list:
    """ Skaber l sin/cos kernels, som kan bruges til at lave cnn learning på.
    Kan plotte hvis man vil:)"""
    transform = transforms.ToTensor()
    n = kernel_size
    angle_map = np.zeros((n,n))
    for x in range(int(-np.floor(n/2)), int(np.ceil(n/2))):    
        for y in (range(int(-np.floor(n/2)), int(np.ceil(n/2)))):

            if x == 0:
                if y > 0:
                    thet = np.pi/2
                elif y < 0:
                    thet = np.pi*3 /2
                else:
                    thet = 0
            else:
                thet = np.arctan(y/x)
                if x < 0:
                    if y < 0:
                        thet -= np.pi
                    else:
                        thet += np.pi
            xx = x + int(np.floor(n/2))
            yy = n - (y + int(np.floor(n/2))) -1

            angle_map[yy,xx] = thet

    lst = []
    kernel_0 = np.ones((n,n))
    lst.append(kernel_0)
    for l_ in range(1,l+1):
        kernel_sin = np.sin(l_*angle_map)
        kernel_cos = np.cos(l_*angle_map)
        kernel_sin = transform(kernel_sin)
        kernel_cos = transform(kernel_cos)
        lst.append(kernel_sin)
        lst.append(kernel_cos)
    if plot:
        fig, axes = plt.subplots(2, l, figsize=(3 * l, 6))
        if l == 1:
            axes = axes.reshape(2, 1)
        for l_ in range(1, l+1):
            axes[0, l_ - 1].imshow(lst[2*l_ - 1].squeeze())
            axes[0, l_ - 1].set_title(f"sin (l={l_})")
            axes[0, l_ - 1].axis('off')
            axes[1, l_ - 1].imshow(lst[2*l_].squeeze())
            axes[1, l_ - 1].set_title(f"cos (l={l_})")
            axes[1, l_ - 1].axis('off')
            
        plt.show()
    return lst


