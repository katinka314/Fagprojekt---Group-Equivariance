import numpy as np
import matplotlib.pyplot as plt
import importlib

import torch
import torch.nn as nn
import torch.nn.functional  as F

from torchvision import transforms

import kagglehub
import pandas as pd
import time





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
    radius_map = radius_map/np.max(radius_map)
    basis = []
    
    for l_ in range(-l,l+1):
        if l_ == 0:
            kernel = transform(np.ones((n,n)))
        else:
            kernel = transform(np.exp(1j* l_*angle_map))
        # kernel_sin = transform(np.sin(l_*angle_map))
        # kernel_cos = transform(np.cos(l_*angle_map))
        basis.append(kernel.to(torch.complex64))
        # basis.append(kernel_sin)
        # basis.append(kernel_cos)
        
    if plot: # plots the non-constant basis functions
        #OBS imshow uses rows downwards. We therefore transpose the matrix before plotting.
        # Complex basis: e^(i*l*theta) = cos(l*theta) + i*sin(l*theta)
        # -> real part is cos, imag part is sin
        fig, axes = plt.subplots(2, l, figsize=(3 * l, 6))
        if l == 1:
            axes = axes.reshape(2, 1)
        for l_ in range(1, l + 1):
            kernel_np = basis[l_].squeeze().numpy().T
            axes[0, l_ - 1].imshow(kernel_np.real)
            axes[0, l_ - 1].set_title(f"cos (l={l_})")
            axes[0, l_ - 1].axis('off')
            axes[1, l_ - 1].imshow(kernel_np.imag)
            axes[1, l_ - 1].set_title(f"sin (l={l_})")
            axes[1, l_ - 1].axis('off')
        plt.show()
    basis = torch.stack(basis)
    basis.squeeze_(1)

    return basis, radius_map


class MLP_Radius(nn.Module):
    def __init__(self, in_features= 1, hidden_units = 16, out_features = 1, bias = True, depth = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_units = hidden_units
        self.depth = depth
        
        layers = []
        layers += [nn.Linear(in_features, hidden_units, bias=bias), nn.ReLU()]
        for _ in range(depth-1):
            layers += [nn.Linear(hidden_units, hidden_units, bias=bias), nn.ReLU()]
        layers += [nn.Linear(hidden_units, out_features, bias=bias)]
        
        self.layer = nn.Sequential(*layers)                 
        
    def forward(self, x):
        return self.layer(x)



class LiftingLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, l, MLP_size: int = 4, bias=True):
        super(LiftingLayer, self).__init__()
        
        # Define learnable parameters
        self.k_size = kernel_size
 
        self.l = l
        self.out_features = out_features
        self.bias = bias
        
        self.basis, self.radius_map = fourier_basis(kernel_size = self.k_size, l = self.l)
    
        self.mlps = nn.ModuleList([MLP_Radius(bias = self.bias, hidden_units=MLP_size) for _ in range(self.out_features*len(self.basis))])
    
    def parsing(self, out):
        
        return
        
         
    def forward(self, x):
        # x shape: (batch_size, in_features)
        # FIX: the dataset yields float32 but the kernels are complex64, and F.conv2d
        # refuses to mix dtypes. Remove these two lines for the old behavior (requires complex input).
        if not x.is_complex():
            x = x.to(torch.complex64)

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
    def __init__(self, in_features, out_features, kernel_size, l,  MLP_size: int = 4, bias=True, n_rings = 4):
        super(ConvLayer, self).__init__()
        
        # Define learnable parameters
        self.k_size = kernel_size
 
        self.l = l
        self.len_basis = l*2 + 1 
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.basis, self.radius_map = fourier_basis(kernel_size = self.k_size, l = self.l)

        #self.mlps = nn.ModuleList([MLP_Radius(bias = self.bias, hidden_units=MLP_size) for _ in range(self.out_features*self.len_basis*self.in_features)])

        ### project gaussian boy
        n_rings = self.k_size // 2 + 2 #husk at argumentere for mængde
        means = torch.linspace(0,1,n_rings,dtype = torch.float32)
        std = (means[1] - means[0])/2 # Distance between 2 centers
        r = torch.tensor(self.radius_map, dtype= torch.float32)
        self.radials = torch.exp(-(r[None] - means[:, None, None])**2 / (2 * std**2))  # (J, k, k)
        # Kaiming-style scaling by 1/sqrt(fan_in): one output pixel is a sum of
        # in_features * k^2 products, so without scaling the activations grow by
        # ~sqrt(fan_in) per layer and the logits end up around 1e8.
        init_scale = (in_features * kernel_size**2) ** -0.5
        self.weights = nn.Parameter(init_scale * torch.randn(n_rings, out_features * self.len_basis * in_features, dtype=torch.float32))
        # OLD version (unscaled init, and hardcoded 4 instead of n_rings, same value for kernel_size=5):
        # self.weights = nn.Parameter(torch.randn(4,out_features * self.len_basis * in_features, dtype=torch.float32))

        # Null kernel: en (l, n, n) tensor af nuller, der matcher basis i dtype.
        # Vi concatenater den på basis-dimensionen, saa self.basis bliver (2l+2, n, n).
        nullkernel = torch.zeros((self.l, self.k_size, self.k_size), dtype=self.basis.dtype)
        self.basis = torch.cat([self.basis, nullkernel], dim=0)

        radius_map = torch.tensor(self.radius_map).float().flatten().unsqueeze(1)

        idxs = []
        #mlp_count = 0
        for channel in range(self.out_features):
            for out_freq in range(-self.l, self.l + 1):
                for in_feature in range(self.in_features):
                    in_freq = in_feature%(self.len_basis) - self.l
                    basis_freq = out_freq - in_freq
                    basis_idx = basis_freq + self.l
                    idxs.append(basis_idx)
                    #MLP_Radius_ = self.mlps[mlp_count]
                    #mlp_count += 1

                    #radial_weights = MLP_Radius_(radius_map).squeeze_().reshape(self.k_size,self.k_size)
                    #kernels.append(self.basis[basis_idx] * radial_weights)
        self.idxs = torch.tensor(idxs)

        # OLD version (kernels built once in __init__). Crashed on the second backward call,
        # because the autograd graph from self.weights was consumed by the first batch:
        # radial_weigths = torch.einsum('cp,chw->hwp', self.weights, self.radials)
        # kernels = torch.zeros(self.k_size,self.k_size, out_features * self.len_basis *self.in_features, dtype = torch.complex64)
        # for basis_idx in range(len(self.basis)):
        #     mask = (self.idxs == basis_idx)
        #     kernels[:,:,mask] = self.basis[basis_idx][:,:,None] * radial_weigths[:,:,mask]
        # self.kernels_tensor = kernels.permute(2,0,1).reshape(self.out_features * (self.len_basis),self.in_features, self.k_size, self.k_size)

    def build_kernels(self):
        # The kernels must be rebuilt on every forward call (not in __init__), so the
        # autograd graph from self.weights is fresh for each batch.
        radial_weigths = torch.einsum('cp,chw->hwp', self.weights, self.radials)
        kernels = torch.zeros(self.k_size, self.k_size, self.out_features * self.len_basis * self.in_features, dtype = torch.complex64)
        for basis_idx in range(len(self.basis)):
            mask = (self.idxs == basis_idx)
            kernels[:,:,mask] = self.basis[basis_idx][:,:,None] * radial_weigths[:,:,mask] # we do [:,:,None] bc otherwise it is just 1 basis, but we want to make sure the dimentions match in broadcasting.

        return kernels.permute(2,0,1).reshape(self.out_features * (self.len_basis),self.in_features, self.k_size, self.k_size) #måske er dim gale pewrmute?

    def forward(self, x):
        # x shape: (batch_size, in_features)

        out = F.conv2d(input = x, weight=self.build_kernels())
        # OLD version: out = F.conv2d(input = x, weight=self.kernels_tensor)

        return out
    
    
class ProjectionLayer(nn.Module):
    def __init__(self, out_features, l, bias = True):
        super().__init__()
        self.l = l
        self.out_features = out_features
        self.bias = bias
        self.len_basis = l*2 + 1
        self.lin = nn.LazyLinear(
                self.out_features,
                bias=self.bias
            )
        
        
    def forward(self,x):
        #x.shape = antal billeder X kanaler X H X W
        x_invariant = torch.abs(x).flatten(start_dim=1) # tager normen og laver til array/flattener
        dummy_x = torch.randn_like(x_invariant)
        self.lin(dummy_x)
        return self.lin(x_invariant)
        #x_invariant = x[:, self.l::self.len_basis,:,:]
        #x_pooled = x_invariant.mean(dim = (2,3))
        #x_pooled = torch.abs(x_pooled).to(torch.float32)
        
        
        out = self.lin.forward(x_invariant)
        return out
"""class GroupEquivariantReLU(nn.Module):
    
    Naive norm-gate baseline. Ækvivariant, men degenereret: tager normen over
    hele (H, W) pr. kanal og nuller kanalen hvis normen er under threshold.
    Global-lineær pr. kanal (ingen lokal selektivitet) og hård tærskel.

    Til en rigtig, eksakt ækvivariant ikke-linearitet: se NormNonlinearity.
    
    def __init__(self, l, n_samples=None, naive=True, threshold=15):
        super().__init__()
        self.l = l
        self.len_basis = 2 * l + 1
        self.naive = naive
        self.threshold = threshold

    def forward(self, x):
        if self.naive:
            e_norm = torch.norm(x, dim=(-2, -1))
            mask = e_norm > self.threshold
            return x * mask[:, :, None, None]

        return x"""
    
class ComplexAdaptiveAvgPool2d(nn.Module):
    """nn.AdaptiveAvgPool2d understoetter ikke complex tensors.
    Pooling er lineaer, saa vi pooler real- og imaginaerdel hver for sig (samme resultat)."""
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)
    def forward(self, x):
        if x.is_complex():
            return torch.complex(self.pool(x.real), self.pool(x.imag))
        return self.pool(x)

class ComplexAdaptiveAvgPool2d(nn.Module):
    """nn.AdaptiveAvgPool2d does not support complex tensors.
    Pooling is linear, so we pool the real and imaginary parts separately (same result)."""
    def __init__(self, output_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        if x.is_complex():
            return torch.complex(self.pool(x.real), self.pool(x.imag))
        return self.pool(x)


class NormNonlinearity(nn.Module):
    """
        f_out = f * relu(|f| - b) / (|f| + eps)
    """
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.bias = nn.Parameter(0.3 * torch.ones(num_channels))  
        self.eps = eps

    def forward(self, x):                       # x: (B, C, H, W) complex
        m = x.abs()                             
        b = self.bias.view(1, -1, 1, 1)
        gate = F.relu(m - b) / (m + self.eps)  
        return x * gate.to(x.dtype)
    



if __name__ == '__main__':
    #path = '/Users/nr1/.cache/kagglehub/datasets/zalando-research/fashionmnist/versions/4'
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    df = pd.read_csv(path + '/fashion-mnist_train.csv')
    num_images = 32
    df_train = df[:num_images]
    rows = df_train.iloc[:, 1:].values
    imgs = rows.reshape(-1, 28,28).astype(np.uint8)
    images = torch.tensor(imgs, dtype=torch.float32) / 255.0   # (1000, 784)
    images = images.unsqueeze_(1).to(torch.complex64)

    
    # Lifting layer to build against
    l = 2
    ll = LiftingLayer(in_features=10, out_features=10, kernel_size=5, l=l)
    conv = ConvLayer(in_features=10*(l*2+1), out_features=12, kernel_size=5, l=l)

    prøy = ProjectionLayer(in_features=12, out_features=10, l=l)
    # Dummy image so we can develop without downloading a dataset.
    # Shape (batch, channels, height, width), complex to match the Fourier basis.

    out = ll.forward(images)
    out2 = conv.forward(out)
    out3 = prøy.forward(out2)
    
    print(out.shape)
    print(out[0][7].shape)