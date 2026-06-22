import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional  as F

from torchvision import transforms

import kagglehub
import pandas as pd


def fourier_basis(kernel_size: int, l: int, plot: bool = False) ->  list:
    """
    Input:
    kernel_size: int: the height/width of the kernel.
    l: the resolution of the basis; the frequency of the highest basis function.
    l = 0 gives the constant function; l = 1 gives constant, cosine and sine with frequency 1.

    Output:
    1 + 2 * l basis functions (kernels) of size n x n
    """
    transform = transforms.ToTensor()
    n = kernel_size
    angle_map = np.zeros((n,n)) # maps each position in kernel to an angle.
    radius_map = np.zeros((n,n)) # maps each position in kernel to an radius
    center_coords = [-(n - 1)/2 + i for i in range(n)] # the x/y coordinates, when origo is set in the middle of the kernel. # n=5: from -2 to 2. n=6: from -2.5 to 2.5
    for x in center_coords:
        for y in center_coords:
            x_idx = int(x + center_coords[-1])
            y_idx = (n-1) - int(y + center_coords[-1])

            theta = np.arctan2(y,x) # arctan2 takes the arguments in this order
            angle_map[x_idx, y_idx] = theta

            r = np.sqrt(x**2 + y**2)
            radius_map[x_idx, y_idx] = r
    radius_map = radius_map/np.max(radius_map)
    basis = []

    for l_ in range(-l,l+1):
        if l_ == 0:
            kernel = transform(np.ones((n,n)))
        else:
            harmonic = (np.exp(1j* l_*angle_map))

            #I center are they undefined, we set them to 0
            harmonic[n // 2, n//2] = 0
            kernel = transform(harmonic)
        basis.append(kernel.to(torch.complex64))


    if plot: # plots the non-constant basis functions
        #OBS imshow uses rows downwards. We therefore transpose the matrix before plotting.
        # Complex basis
        #real part is cos, imag part is sin
        fig, axes = plt.subplots(2, l, figsize=(3 * l, 6))
        if l == 1:
            axes = axes.reshape(2, 1)
        for l_ in range(1, l + 1):
            kernel_np = basis[l + l_].squeeze().numpy().T
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

        basis, self.radius_map = fourier_basis(kernel_size=self.k_size, l=self.l) #Initializing the basis
        self.register_buffer("basis", basis, persistent=False) 
        radius_col = torch.tensor(self.radius_map, dtype=torch.float32).flatten().unsqueeze(1)
        self.register_buffer("radius_col", radius_col, persistent=False)

        self.mlps = nn.ModuleList([MLP_Radius(bias=self.bias, hidden_units=MLP_size)
                                   for _ in range(self.out_features * len(basis))])

    def forward(self, x):
        if not x.is_complex():
            x = x.to(torch.complex64)
        kernels = []
        for j in range(self.out_features):
            for i in range(len(self.basis)):
                mlp = self.mlps[j * len(self.basis) + i]
                radial_weights = mlp(self.radius_col).squeeze_().reshape(self.k_size, self.k_size)
                kernels.append(self.basis[i] * radial_weights)
        kernels = torch.stack(kernels).unsqueeze_(1)
        return F.conv2d(input=x, weight=kernels)



class ConvLayer(nn.Module):
    """ Convolutional layer for model uses gaussian rings instead og MLP's for radial part
        
        Builds the convolutional layer to preserve invariance
        
        """
    def __init__(self, in_features, out_features, kernel_size, l,  MLP_size: int = 4, bias=True, n_rings = None):
        super(ConvLayer, self).__init__()

        # Define learnable parameters
        self.k_size = kernel_size

        self.l = l
        self.len_basis = l*2 + 1
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        basis, self.radius_map = fourier_basis(kernel_size = self.k_size, l = self.l)

        # Gaussian-ring radial basis
        n_rings = self.k_size // 2 + 2 if n_rings == None else n_rings
        means = torch.linspace(0,1,n_rings,dtype = torch.float32)
        std = (means[1] - means[0])/2 # Distance between 2 centers
        r = torch.tensor(self.radius_map, dtype= torch.float32)
        radials = torch.exp(-(r[None] - means[:, None, None])**2 / (2 * std**2))  # (J, k, k)
        # Scaling by 1/sqrt(in_features x kernel_size^2) used to control logit output.
        init_scale = (in_features * kernel_size**2) ** -0.5
        self.weights = nn.Parameter(init_scale * torch.randn(n_rings, out_features * self.len_basis * in_features, dtype=torch.float32))

        # Null kernel: an (l, n, n) tensor of zeros matching the basis dtype.
        nullkernel = torch.zeros((self.l, self.k_size, self.k_size), dtype=basis.dtype)
        basis = torch.cat([basis, nullkernel], dim=0)

        idxs = []
        for channel in range(self.out_features):
            for out_freq in range(-self.l, self.l + 1):
                for in_feature in range(self.in_features):
                    in_freq = in_feature % self.len_basis - self.l
                    basis_freq = out_freq - in_freq
                    basis_idx = basis_freq + self.l
                    idxs.append(basis_idx)
        idxs = torch.tensor(idxs)

        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer("radials", radials, persistent=False)
        self.register_buffer("idxs", idxs, persistent=False)
        self._kernel_cache = None


    def build_kernels(self):
        radial_weights = torch.einsum('cp,chw->hwp', self.weights, self.radials)
        radial_weights = radial_weights.permute(2, 0, 1)
        kernels = self.basis[self.idxs] * radial_weights
        return kernels.reshape(self.out_features * self.len_basis,
                               self.in_features, self.k_size, self.k_size)

    def forward(self, x):
        if self.training:
            kernels = self.build_kernels()
        else:
            if self._kernel_cache is None or self._kernel_cache.device != self.weights.device:
                self._kernel_cache = self.build_kernels().detach()
            kernels = self._kernel_cache
        return F.conv2d(input=x, weight=kernels)

    def train(self, mode: bool = True):
        if mode:
            self._kernel_cache = None
        return super().train(mode)


class ProjectionLayer(nn.Module):
    """ Projection layer for final output of logit values """
    def __init__(self, out_features, l, bias = True):
        super().__init__()
        self.l = l
        self.out_features = out_features
        self.bias = bias
        self.len_basis = l*2 + 1
        self.lin = nn.LazyLinear(self.out_features, bias=self.bias)
        self.first = True


    def forward(self,x):

        # x.shape = num images x channels x H x W
        _abs_x = torch.abs(x) # Converting to real
        x_mean = _abs_x.mean(axis = (2,3)) # mean-pool instead of a linear flatten(start_dim=1) to preserve invariance
        x_max = _abs_x.amax(dim = (2,3)) # max

        x_std = _abs_x.std(axis = (2,3)) # std for invariance
        x_invariant = torch.cat((x_mean,x_max,x_std), dim = 1).flatten(start_dim=1) # Joining all invaraint aspects

        if self.first: # Dummy control, to initialize lazy layer.
            dummy_x = torch.randn_like(x_invariant)
            self.lin(dummy_x)
            self.first = False
        return self.lin(x_invariant)



class ComplexAdaptiveAvgPool2d(nn.Module):
    """nn.AdaptiveAvgPool2d does not support complex tensors.
    Pooling is linear, so we pool the real and imaginary parts separately"""
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
    path = kagglehub.dataset_download("zalando-research/fashionmnist")
    df = pd.read_csv(path + '/fashion-mnist_train.csv')
    num_images = 32
    df_train = df[:num_images]
    rows = df_train.iloc[:, 1:].values
    imgs = rows.reshape(-1, 28,28).astype(np.uint8)
    images = torch.tensor(imgs, dtype=torch.float32) / 255.0
    images = images.unsqueeze_(1).to(torch.complex64)

    # Build one of each layer and check the projection is invariant under a 90-degree rotation.
    l = 2
    ll = LiftingLayer(in_features=10, out_features=10, kernel_size=5, l=l)
    conv = ConvLayer(in_features=10*(l*2+1), out_features=12, kernel_size=5, l=l)
    proj = ProjectionLayer(out_features=10, l=l)

    out = ll.forward(images)
    out2 = conv.forward(out)
    out3 = proj.forward(out2)
    out4 = proj.forward(torch.rot90(out2, 1, dims =(2,3)))

    print(out3)
    print(out4)
