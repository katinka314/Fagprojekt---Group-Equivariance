
import numpy as np
import matplotlib.pyplot as plt

#import torch 
#import torch.nn as nn
#import torch.nn.functional  as F
from torchvision import transforms
#from torch import Tensor

#import kagglehub
#import pandas as pd
#import PIL.Image as Image


""" TODO: Vi skal skrive den med underfunktioner for læsbarhed """
def fourier_basis(kernel_size: int, l: int, plot: bool = False) ->  list:
    """ 
    Input:
    kernel_size: int: højden/bredden på kernel. 
    l: hvor høj resulution af basiser du får. er frekvensen af højeste basis.
    l = 0 giver den konstante funktion ud, l = 1 giver konstant, cosinus og sinus med frkvens = 1
        
    Output:
    1 + 2 * l basisfunktioner (kernels) i størrelse nxn
    
    """
    transform = transforms.ToTensor()
    n = kernel_size
    angle_map = np.zeros((n,n)) # maps each position in kernel to an angle.
    radius_map = np.zeros((n,n)) # maps each position in kernel to an radius
    center_coords = [-(n - 1)/2 + i for i in range(n)] # the x/y coordinates, when origo is set in the middle of the kernel. #n = 5: fra -2 til 2. n = 6 fra -2,5 til 2,5
    for x in center_coords:    
        for y in center_coords:
            x_idx = int(x + center_coords[-1])
            y_idx = (n-1) - int(y + center_coords[-1])

            theta = np.arctan2(y,x) #functionen tagerargumenterne ind i den rækkefølge somehow.
            angle_map[x_idx, y_idx] = theta

            r = np.sqrt(x**2 + y**2)
            radius_map[x_idx, y_idx] = r
    basis = []
    kernel_0 = transform(np.ones((n,n)))
    basis.append(kernel_0)
    for l_ in range(1,l+1):
        kernel_sin = transform(np.sin(l_*angle_map))
        kernel_cos = transform(np.cos(l_*angle_map))
        basis.append(kernel_sin)
        basis.append(kernel_cos)
        
    if plot: # plots the non-constant basis functions
        #OBS imshow uses rows downwards. We therefore transpose the matrix before plotting.
        fig, axes = plt.subplots(2, l, figsize=(3 * l, 6))
        if l == 1:
            axes = axes.reshape(2, 1)
        for l_ in range(1, l+1):
            axes[0, l_ - 1].imshow(basis[2*l_ - 1].squeeze().T)
            axes[0, l_ - 1].set_title(f"sin (l={l_})")
            axes[0, l_ - 1].axis('off')
            axes[1, l_ - 1].imshow(basis[2*l_].squeeze().T)
            axes[1, l_ - 1].set_title(f"cos (l={l_})")
            axes[1, l_ - 1].axis('off')
        plt.show()

    return basis, radius_map


