import os
print(os.getcwd())

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import importlib

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from models.NN_layers import *
from models.Model import *

import torch 
import torch.nn as nn
import torch.nn.functional  as F

from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode

import pandas as pd
import PIL.Image as Image

import kagglehub
path = kagglehub.dataset_download("zalando-research/fashionmnist")


def format_img(num_images = 1000):
    df_train = df[:num_images]
    img_rows = df_train.iloc[:, 1:].to_numpy() #converts to numpy array (outer dim is pictures)
    img_square = img_rows.reshape(-1, 28,28).astype(np.uint8) # reshape inner dim to be a pictur HxW
    images = torch.tensor(img_square, dtype=torch.float32).unsqueeze_(1) / 255.0  # make into tensor and scale pixel values to be in range [0,1] instead of [0,255]
    return images

def rotate_batch(images, angles_deg):
    # images: [B, C, H, W]
    B, C, H, W = images.shape
    device = images.device

    out = []

    for angle in angles_deg:
        theta = np.radians(angle)

        # rotation matrix (inverse mapping for grid_sample)
        rot = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0]
        ], dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(rot, images.size(), align_corners=False)
        rotated = F.grid_sample(images, grid, align_corners=False)
        out.append(rotated)
    return torch.stack(out, dim=1)  # [B, 8, C, H, W]

def plot_feature_grid(x_list, channel):
    """
    x_list: list of tensors, each [Channels, H, W] or [1, Channels, H, W]
    """
    
    x_list = [t.squeeze(0) if t.dim() == 4 and t.shape[0] == 1 else t 
        for t in x_list] #fix dimentions if they are not correct - End up with [Channels, H, W]

    org_img = x_list[0][channel] #extract non-rotated image
    #org_img_rotations = rotate_batch(org_img[None,None], angles_deg=angles).squeeze(0).squeeze(1).detach().numpy() #rotate the non-rotated (so that we can compare it to the other rotations.)
    rotations = []

    for angle in angles:
        rotated = rotate(
            org_img.unsqueeze(0),  # [1, H, W]
            angle=angle,
            fill=0,
            interpolation=InterpolationMode.BILINEAR
        )
        rotations.append(rotated.squeeze(0))

    org_img_rotations = torch.stack(rotations)  # [num_angles, H, W]    
    
    n = len(x_list)
    fig, axes = plt.subplots(2, n, figsize=(3*n, 6))

    for i, (x,org) in enumerate(zip(x_list,org_img_rotations)):
        img = x[channel].detach().cpu()  # extract 1 channel -> [H, W]

        #plot the image that was not rotated before being parsed to the model and aftewards rotated theta degrees
        axes[0, i].imshow(org.detach().numpy(), cmap='gray')
        axes[0, i].axis("off")
        axes[0, i].set_title(f"rho(g) * f(x), theta:{angles[i]} ")

        # plot the image that were rotated before being parsed to the model
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis("off")
        axes[1, i].set_title(f"f(rho(gx)), theta: {angles[i]} ")

    plt.tight_layout()
    plt.show()

#LOAD MODEL WEIGHTS ========================================================
model_weights_path = Path(__file__).resolve().parents[1] / "models" / "model_weights" / "GE_CNN_model_weights_.pth"
model_weights = torch.load(model_weights_path)
#model_weights = torch.load("models/model_weights/CNN_model_weights_.pth")

print("Loaded model:", model_weights["model_name"])

if model_weights["model_name"] == "CNN":
    model = CNN(**model_weights["model_args"])
elif model_weights["model_name"] == "GE_CNN":
    model = GE_CNN(**model_weights["model_args"])
model.name = model_weights["model_name"]

model.load_state_dict(model_weights["state_dict"])
print("---Loaded model---")

# MAKE PARTIAL MODEL (only layers up until some target layer) ================
if model.name == "CNN":
    layers = list(model.features.children())
if model.name == "GE_CNN":
    layers = list(model.model.children()) #get all layers in model
target_layer = min(1, len(layers)) #choose target layer
partial_model = nn.Sequential(*layers[:target_layer])
print("Partial model:")
print(partial_model)

# LOAD BILLEDER ==============================================================
df = pd.read_csv(path + '/fashion-mnist_train.csv') #læs billeder
images = format_img(num_images=3) 
print("---Loaded images---")

# ROTER BILLEDER =============================================================
n_rotations = 8
angles = [i * 45 for i in range(n_rotations)]
#roterer de 3 billeder
rotated_images = rotate_batch(images, angles) # shape N_images X N_rotations X 1 X H X W


# PLOT parwise comparison ======================================================
for image in rotated_images:
    features = []
    for rotation in image:
        intermediate = partial_model(rotation) # get output of partial model 
        features.append(abs(intermediate))
    
    plot_feature_grid(features, channel = 0) #choose which channel to plot
