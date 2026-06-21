import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

fsize = 30

from models.NN_layers import *
torch.manual_seed(1)
l = 2
basis, radius_map = fourier_basis(kernel_size = 100, l = l, plot = False)

# CIRCULAR BASIS
fig, axes = plt.subplots(2, l, figsize=(3 * l, 6))
if l == 1:
    axes = axes.reshape(2, 1)
for l_ in range(1, l + 1):
    kernel_np = basis[l + l_].squeeze().numpy().T 
    axes[0, l_ - 1].imshow(kernel_np.real)
    axes[0, l_ - 1].set_title(f"real part (l={l_})")
    axes[0, l_ - 1].axis('off')
    axes[1, l_ - 1].imshow(kernel_np.imag)
    axes[1, l_ - 1].set_title(f"imag part (l={l_})")
    axes[1, l_ - 1].axis('off')
save_path = os.path.join(FAG_PROJEKT_DIR, "reports", "plots", "basis",f"circular_basis.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

# CONSTRUCT THE RADIAL BASIS
n_rings = 5
means = torch.linspace(0,1,n_rings,dtype = torch.float32)
std = (means[1] - means[0])/2 # Distance between 2 centers
r = torch.tensor(radius_map, dtype= torch.float32)
radials = torch.exp(-(r[None] - means[:, None, None])**2 / (2 * std**2))  # (J, k, k)
weights = torch.randn((len(basis), n_rings), dtype = torch.float32)*2 - 1
radial_weights = torch.einsum('nc,chw->nhw', weights, radials)

# PLOT SUMMED KERNEL
plt.imshow(radial_weights[0])
save_path = os.path.join(FAG_PROJEKT_DIR, "reports", "plots", "basis",f"radial_kernel.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

# PLOT RADIAL BASIS
fig, axes = plt.subplots(1, n_rings + 1, figsize=(18, 3))

for i in range(n_rings):
    axes[i].imshow(radials[i].detach().numpy())
    axes[i].set_title(rf"$\phi_{i}$", fontsize=fsize)
    axes[i].axis("off")
axes[n_rings].imshow(radial_weights[0])
axes[n_rings].set_title(rf"$K^{{\rightarrow}}(r|w)$",  fontsize=fsize)
axes[n_rings].axis("off")

plt.tight_layout()
save_path = os.path.join(FAG_PROJEKT_DIR, "reports", "plots", "basis",f"radial_basis.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()

# PLOT FINAL KERNELS
fig, axes = plt.subplots(1, len(basis), figsize=(15, 3))
for i in range(len(basis)):
    axes[i].imshow(basis[i].squeeze().numpy().T.real * radial_weights[i].detach().numpy())
    axes[i].set_title(f"l = {i-l}",  fontsize=fsize)
    axes[i].axis("off")

plt.tight_layout()
save_path = os.path.join(FAG_PROJEKT_DIR, "reports", "plots", "basis",f"kernels.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close()