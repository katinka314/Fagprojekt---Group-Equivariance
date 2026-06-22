import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional  as F

from torch import Tensor
from torchvision import transforms

from models.NN_layers import *

class GE_CNN(nn.Module):
    """GE-CNN model the model takes as an input:
    Kernel_siz, l (n_harmonics), in_features, img_size, MLP_Size (deprecated)
    n_convolutional layers (normally unused), conv_pr_pool (unedited)
    n_rings, in alle experiments set to 6
    """
    def __init__(self, kernel_size: int, l: int, in_features: int = 1, img_size: int = 28, MLP_size: int = 4, n_conv_layers: int = 3, conv_pr_pool: int = 2, channels: int = 16, n_classes:int = 10, bias:bool = True, n_rings = 6):
        super().__init__()

        self.n_classes = n_classes
        self.len_basis = 2 * l + 1

        self.layer_sizes = [2**i for i in range(int(np.log2(channels)), int(np.log2(channels)) + n_conv_layers + 1)]
        ls = self.layer_sizes

        layers = []
        for i in range(n_conv_layers):
            layers.append(ConvLayer(ls[i]*self.len_basis, ls[i+1], kernel_size=kernel_size, l=l, bias=bias, n_rings=n_rings))
            for j in range(conv_pr_pool - 1):
                layers.append(ConvLayer(ls[i+1]*self.len_basis, ls[i+1], kernel_size=kernel_size, l=l, bias=bias, n_rings=n_rings))

            layers.append(NormNonlinearity(ls[i+1]*self.len_basis))
            reduced_size = img_size//(2**(i+1))
            layers.append(ComplexAdaptiveAvgPool2d((reduced_size, reduced_size)))
            # OLD version: layers.append(nn.AdaptiveAvgPool2d((reduced_size, reduced_size)))  # crashes on complex tensors

        self.model = nn.Sequential(
            LiftingLayer(in_features, ls[0], kernel_size=kernel_size, l=l, bias=bias),
            *layers,
            ProjectionLayer(out_features=n_classes, l=l, bias=bias),

    )



    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    """Standard (non-equivariant) CNN baseline that mirrors GE_CNN's structure:
    same channel progression, same number of conv layers (without padding), one
    nonlinearity per block and the same pooling."""

    def __init__(self, kernel_size: int = 5, in_features: int = 1, img_size: int = 28,  n_conv_layers: int = 3, conv_pr_pool: int = 2, channels: int = 16, n_classes:int = 10, bias:bool = True):
        super().__init__()

        self.layer_sizes = [2**i for i in range(int(np.log2(channels)), int(np.log2(channels)) + n_conv_layers + 1)]
        ls = self.layer_sizes

        # Mirrors the LiftingLayer: first convolution into the base channel width.
        layers = [nn.Conv2d(in_features, ls[0], kernel_size, bias=bias)]
        for i in range(n_conv_layers):
            layers.append(nn.Conv2d(ls[i], ls[i+1], kernel_size, bias=bias))
            for j in range(conv_pr_pool - 1):
                layers.append(nn.Conv2d(ls[i+1], ls[i+1], kernel_size, bias=bias))

            layers.append(nn.ReLU())
            reduced_size = img_size//(2**(i+1))
            layers.append(nn.AdaptiveAvgPool2d((reduced_size, reduced_size)))

        self.features = nn.Sequential(*layers)
        # Mirrors the ProjectionLayer: global pooling + linear layer.
        # self.classifier = nn.Linear(ls[n_conv_layers], n_classes, bias=bias)
        self.n_classes = n_classes
        self.bias = bias
        self.classifier = nn.LazyLinear(self.n_classes, bias = self.bias)


    def forward(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)
