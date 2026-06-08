import numpy as np
import matplotlib.pyplot as plt
import importlib

import functions
importlib.reload(functions)
from functions import *

import torch
import torch.nn as nn
import torch.nn.functional  as F


import kagglehub
import pandas as pd


class MLP_Radius(nn.Module):
    def __init__(self, in_features= 1, hidden_units = 16, out_features = 1, bias = False, depth = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_units = hidden_units
        self.depth = depth
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.bias = None
        
        layers = []
        layers += [nn.Linear(in_features, hidden_units, bias=bias), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden_units, hidden_units, bias=bias), nn.ReLU()]
        layers += [nn.Linear(hidden_units, out_features, bias=bias)]
        
        self.layer = nn.Sequential(*layers)                 
        
    def forward(self, x):
        return self.layer(x)



class LiftingLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, l,  bias=True):
        super(LiftingLayer, self).__init__()
        
        # Define learnable parameters
        self.k_size = kernel_size
 
        self.l = l
        self.out_features = out_features
        self.bias = bias
        
        self.basis, self.radius_map = fourier_basis(kernel_size = self.k_size, l = self.l)
    
        self.mlps = nn.ModuleList([MLP_Radius(bias = self.bias) for _ in range(self.out_features*len(self.basis))])
    
    def parsing(self, out):
        
        return
        
         
    def forward(self, x):
        # x shape: (batch_size, in_features)
       
        radius_map = torch.tensor(self.radius_map).float().flatten().unsqueeze(1)
        kernels = []
        for j in range(self.out_features):
            for i in range(len(self.basis)):
                MLP_Radius_ = self.mlps[j*len(self.basis)+i]
                radial_weights = MLP_Radius_(radius_map).squeeze_().reshape(self.k_size,self.k_size)
                kernels.append(self.basis[i] * radial_weights)
        kernels = torch.stack(kernels).unsqueeze_(1)

        out = F.conv2d(input = x, weight=kernels) #??? SKAL vi specificerer bias, stride padding???

        return out


    
class ConvLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, l,  bias=True):
        super(ConvLayer, self).__init__()
        
        # Define learnable parameters
        self.k_size = kernel_size
 
        self.l = l
        self.len_basis = l*2 + 1 
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.basis, self.radius_map = fourier_basis(kernel_size = self.k_size, l = self.l)
        self.mlps = nn.ModuleList([MLP_Radius(bias = self.bias) for _ in range(self.out_features*self.len_basis*self.in_features)])

        # Null kernel: en (l, n, n) tensor af nuller, der matcher basis i dtype.
        # Vi concatenater den på basis-dimensionen, saa self.basis bliver (2l+2, n, n).
        nullkernel = torch.zeros((self.l, self.k_size, self.k_size), dtype=self.basis.dtype)
        self.basis = torch.cat([self.basis, nullkernel], dim=0)
        self.frequency_dict = {(in_freq, out_freq): [MLP_Radius(bias = self.bias) for _ in range(out_features)] 
                               for in_freq in range(-l, l + 1) 
                               for out_freq in range(-l, l + 1)
                               }
        
        self.basis
    def forward(self, x):
        # x shape: (batch_size, in_features)
        temp_counter_check = []
        radius_map = torch.tensor(self.radius_map).float().flatten().unsqueeze(1)
        
        kernels = []
        for _ in range(self.out_features):
            temp_counter_check.append(f'channel: {_}')
            for out_freq in range(-self.l, self.l + 1):
                temp_counter_check.append(f'out_freq: {out_freq}')
                for in_feature in range(self.in_features):
                    in_freq = in_feature%(self.len_basis) - self.l
                    basis_freq = out_freq - in_freq
                    basis_idx = basis_freq + self.l
                    MLP_Radius_ = MLP_Radius(bias = self.bias)
                    radial_weights = MLP_Radius_(radius_map).squeeze_().reshape(self.k_size,self.k_size)
                    temp_counter_check.append(basis_idx)
                    kernels.append(self.basis[basis_idx] * radial_weights)
                
        
        """
        for in_freq in range(-self.l, self.l + 1):
                MLP_Radius_ = self.frequency_dict[(in_freq, out_freq)]
                radial_weights = MLP_Radius_(radius_map).squeeze_().reshape(self.k_size,self.k_size)
                kernels.append(self.basis[in_freq] * radial_weights)
        kernels = torch.stack(kernels).unsqueeze_(1)
        """
        kernels_tensor = torch.stack(kernels).reshape(self.out_features * (self.len_basis),self.in_features, self.k_size, self.k_size)
        out = F.conv2d(input = x, weight=kernels_tensor) #??? SKAL vi specificerer bias, stride padding???

        return out
    
    

    
    

if __name__ == '__main__':
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    df = pd.read_csv(path + '/fashion-mnist_train.csv')
    num_images = 50
    df_train = df[:num_images]
    rows = df_train.iloc[:, 1:].values
    imgs = rows.reshape(-1, 28,28).astype(np.uint8)
    images = torch.tensor(imgs, dtype=torch.float32) / 255.0   # (1000, 784)
    images = images.unsqueeze_(1).to(torch.complex128)

    
    # Lifting layer to build against
    l = 2
    ll = LiftingLayer(in_features=10, out_features=10, kernel_size=5, l=l)
    conv = ConvLayer(in_features=10*(l*2+1), out_features=10, kernel_size=5, l=l)

    # Dummy image so we can develop without downloading a dataset.
    # Shape (batch, channels, height, width), complex to match the Fourier basis.

    out = ll.forward(images)
    out2 = conv.forward(out)
    print(out.shape)
    print(out[0][7].shape)