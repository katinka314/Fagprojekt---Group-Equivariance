import sys
from pathlib import Path

# Find repo-roden (mappen der indeholder 'fag_projekt/') og laeg den paa sys.path,
# saa 'import fag_projekt...' virker uanset hvor kernen er startet.


import numpy as np
import matplotlib.pyplot as plt
import importlib

import os
os.chdir("..")

import models.NN_layers as NN_layers
importlib.reload(NN_layers)
from models.NN_layers import *

import torch 
import torch.nn as nn
import torch.nn.functional  as F

from torch import Tensor
from torchvision import transforms

class NaiveGE_CNN(nn.Module):
    def __init__(self, kernel_size: int, l: int, in_features: int = 1, img_size: int = 28, MLP_size: int = 4, n_conv_layers: int = 3, conv_pr_pool: int = 2, channels: int = 16, n_classes:int = 10, bias:bool = True):
        super().__init__()
        
        """self.kernel_size = kernel_size
        self.l = l
        self.in_features = in_features
        self.n_conv_layers = n_conv_layers
        self.bias = bias"""
        
        self.n_classes = n_classes
        self.len_basis = 2 * l + 1

        self.layer_sizes = [2**i for i in range(int(np.log2(channels)), int(np.log2(channels)) + n_conv_layers + 1)]
        ls = self.layer_sizes

        layers = []
        for i in range(n_conv_layers):
            layers.append(ConvLayer(ls[i]*self.len_basis, ls[i+1], kernel_size=kernel_size, l=l, bias=bias))
            for j in range(conv_pr_pool - 1):
                layers.append(ConvLayer(ls[i+1]*self.len_basis, ls[i+1], kernel_size=kernel_size, l=l, bias=bias))
        
            layers.append(NormNonlinearity(ls[i+1]*self.len_basis))
            reduced_size = img_size//(2**(i+1))
            layers.append(ComplexAdaptiveAvgPool2d((reduced_size, reduced_size)))

        self.model = nn.Sequential(
            LiftingLayer(in_features, ls[0], kernel_size=kernel_size, l=l, bias=bias),
            *layers,
            ProjectionLayer(ls[n_conv_layers], n_classes, l=l, bias=bias),
    )
    
    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self, kernel_size: int = 5, l: int = 2, in_features: int = 1, img_size: int = 28, MLP_size: int = 4, n_conv_layers: int = 3, conv_pr_pool: int = 2, channels: int = 16, n_classes:int = 10, bias:bool = True):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
