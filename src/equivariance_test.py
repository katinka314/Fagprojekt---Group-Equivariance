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

from data import RotatedMNIST

import torch 
import torch.nn as nn
import torch.nn.functional  as F

from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from train import evaluate

import pandas as pd
import PIL.Image as Image

import kagglehub
path = kagglehub.dataset_download("zalando-research/fashionmnist")

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
WEIGHT_FILE = "GE_CNN_weights_seed4.pth" #choose something from model/model_weights folder
TARGET_LAYER = 5
NUM_IMAGES = 3
N_ROTATIONS = 9
ANGLES = [i * 22.5 for i in range(N_ROTATIONS)]


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

def plot_feature_grid(x_list, channel, model, save_idx):
    """
    x_list: list of tensors, each [Channels, H, W] or [1, Channels, H, W]
    """
    
    x_list = [t.squeeze(0) if t.dim() == 4 and t.shape[0] == 1 else t 
        for t in x_list] #fix dimentions if they are not correct - End up with [Channels, H, W]

    org_img = x_list[0][channel] #extract non-rotated image
    #org_img_rotations = rotate_batch(org_img[None,None], angles_deg=angles).squeeze(0).squeeze(1).detach().numpy() #rotate the non-rotated (so that we can compare it to the other rotations.)
    rotations = []

    for angle in ANGLES:
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
        axes[0, i].set_title(f"rho(g) * f(x), theta:{ANGLES[i]} ")

        # plot the image that were rotated before being parsed to the model
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].axis("off")
        axes[1, i].set_title(f"f(rho(gx)), theta: {ANGLES[i]} ")

    plt.tight_layout()
    save_path = os.path.join(FAG_PROJEKT_DIR, "reports", "plots", "equivariance_test",f"{model}_feature_plot_img{save_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_class_scores(tensors, model, save_idx):
    # [num_angles, num_classes]
    data = torch.stack(tensors).detach().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(data, aspect='auto', cmap='viridis')

    plt.xlabel("Class")
    plt.ylabel("Image rotation in degrees")
    plt.colorbar(label="Value")
    class_names = ["T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]
    plt.xticks(ticks = range(10), labels = class_names)
    plt.yticks(ticks = range(len(tensors)), labels = ANGLES)
    
    save_path = os.path.join(FAG_PROJEKT_DIR, "reports", "plots", "equivariance_test",f"{model}_classscores_img{save_idx}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

#LOAD MODEL WEIGHTS ========================================================
model_weights_path = Path(__file__).resolve().parents[1] / "models" / "model_weights" / WEIGHT_FILE
model_weights = torch.load(model_weights_path, map_location=torch.device('cpu'))
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
target_layer = min(TARGET_LAYER, len(layers)) #choose target layer
partial_model = nn.Sequential(*layers[:target_layer])

print("Partial model:")
print(partial_model)

# LOAD BILLEDER ==============================================================
data = RotatedMNIST(data_dir = os.path.join(FAG_PROJEKT_DIR, "data", "processed"), split = "train", rotated = False, padding = True)
images = data[:NUM_IMAGES][0].permute([1,0,2,3]) #shape [num_img, 1, H, W]

print("---Loaded images---")

# ROTER BILLEDER =============================================================
rotated_images = rotate_batch(images, ANGLES) # shape N_images X N_rotations X 1 X H X W


# PLOT pairwise comparison ======================================================
for i, image in enumerate(rotated_images):
    features = []
    class_scores = []
    for rotation_img in image:
        intermediate_featuremap = partial_model(rotation_img) # get output of partial model
        prediction = model(rotation_img[None])[0]
        features.append(abs(intermediate_featuremap))
        class_scores.append(prediction)
    
    plot_feature_grid(features, channel = 0, model = model.name, save_idx=i) #choose which channel to plot
    plot_class_scores(class_scores, model = model.name, save_idx = i)

    BATCH_SIZE     = 128
    TEST_FRACTION  = 0.1
    PADDING = True
    test_rot_dataset = RotatedMNIST(split="test",  rotated=True,  fraction=TEST_FRACTION, padding=PADDING)
    test_rot_loader = DataLoader(test_rot_dataset, batch_size=BATCH_SIZE, shuffle=False)
    rot_loss, rot_acc = evaluate(model, test_rot_loader, show_progress=False)
    print(rot_acc)