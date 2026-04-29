import numpy as np
import matplotlib.pyplot as plt
import importlib

import functions
importlib.reload(functions)
from functions import *

import torch.nn as nn
import torch.nn.functional  as F


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
        self.MLP_Radius_ = MLP_Radius(bias = bias)
        # Define learnable parameters
        self.k_size = kernel_size
 
        self.l = l
        self.out_features = out_features
        
    def forward(self, x):
        # x shape: (batch_size, in_features)
        #fourier folder
        basis, radius_map = fourier_basis(kernel_size = self.k_size, l = self.l)
        radius_map = torch.tensor(radius_map).float().flatten().unsqueeze(1)
        kernels = []
        for _ in range(self.out_features):
            for i in range(len(basis)):
                radial_weights = self.MLP_Radius_(radius_map).squeeze_().reshape(self.k_size,self.k_size)
                kernels.append(basis[i] * radial_weights)
        kernels = torch.stack(kernels).unsqueeze_(1)

        out = F.conv2d(input = x, weight=kernels) #??? SKAL vi specificerer bias, stride padding???
        
        return out