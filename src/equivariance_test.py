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

import pandas as pd
import PIL.Image as Image

import kagglehub
path = kagglehub.dataset_download("zalando-research/fashionmnist")

#LOAD MODEL WEIGHTS ========================================================
model_weights_path = Path(__file__).resolve().parents[1] / "models" / "model_weights" / "CNN_model_weights_.pth"
model_weights = torch.load(model_weights_path)
#model_weights = torch.load("models/model_weights/CNN_model_weights_.pth")

print(model_weights["model_name"])

if model_weights["model_name"] == "CNN":
    model = CNN(**model_weights["model_args"])
elif model_weights["model_name"] == "GE_CNN":
    model = GE_CNN(**model_weights["model_args"])

model.load_state_dict(model_weights["state_dict"])

# LOAD BILLEDER ==============================================================

def format_img(num_images = 1000):
    df_train = df[:num_images]
    img_rows = df_train.iloc[:, 1:].to_numpy() #converts to numpy array (outer dim is pictures)
    img_square = img_rows.reshape(-1, 28,28).astype(np.uint8) # reshape inner dim to be a pictur HxW
    images = torch.tensor(img_square, dtype=torch.float64).unsqueeze_(1) / 255.0  # make into tensor and scale pixel values to be in range [0,1] instead of [0,255]
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
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0]
        ], device=device).unsqueeze(0).repeat(B, 1, 1)

        grid = F.affine_grid(rot, images.size(), align_corners=False)
        rotated = F.grid_sample(images, grid, align_corners=False)
        out.append(rotated)
    return torch.stack(out, dim=1)  # [B, 8, C, H, W]

df = pd.read_csv(path + '/fashion-mnist_train.csv') #læs billeder
images = format_img(num_images=3)

angles = [i * 45 for i in range(8)]

rotated_images = rotate_batch(images, angles) # shape N_images X N_rotations X 1 X H X W
print(rotated_images.shape)

x = rotated_images[0,0][None] #1 image in the right format (shape [1,1,28,28])
out = model.forward(x)