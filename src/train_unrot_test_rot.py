from data import RotatedMNIST
from train import *
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path
import matplotlib.pyplot as plt

# Change import path to a level up (from src -> Fagprojekt) so 'models' can be imported
FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from models.Model import *

# Fraction of the dataset used for training/evaluation (1.0 = everything).
TRAIN_FRACTION = 0.1
TEST_FRACTION = 0.1
N_EPOCHS = 5

train_dataset = RotatedMNIST(split="train", rotated=False,  fraction=TRAIN_FRACTION)
test_dataset = RotatedMNIST(split="test", rotated= True, fraction=TEST_FRACTION)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# n_conv_layers=3 does not work with kernel_size=5 on 28x28: the feature maps
# shrink to 3x3 before the last block (convs have no padding), too small for a 5x5 kernel.
GE_CNN_args = {
    "kernel_size": 5,
    "l": 1,
    "in_features": 1,
    "img_size": 28,
    "n_conv_layers": 2,
    "conv_pr_pool": 1,
    "channels": 8,
    "n_classes": 10,
    "bias": True,
}
GE_CNN_model = GE_CNN(**GE_CNN_args)
GE_CNN_model.name = "GE_CNN"

CNN_args = {
    "kernel_size": 5,
    "in_features": 1,
    "img_size": 28,
    "n_conv_layers": 2,
    "conv_pr_pool": 1,
    "channels": 8,
    "n_classes": 10,
    "bias": True,
}
CNN_model = CNN(**CNN_args)
CNN_model.name = "CNN"
model_args = {CNN_model: CNN_args, GE_CNN_model: GE_CNN_args}

dummy = torch.zeros(1,1,28,28)
models = {"CNN": CNN_model, "GE_CNN": GE_CNN_model}
for m in models.values():
    m(dummy)

histories = {}
for name, model in models.items():
    print(f"Training {name} ({sum(p.numel() for p in model.parameters())} parameters)")
    histories[name] = train_loop(model, lr=1e-3, train_loader=train_loader,
                                 n_epochs=N_EPOCHS, test_loader=test_loader,
                                 show_progress=False)
    
    #SAVE the model weights =============================================================
    model_specifications = "" # string indicating parameters of model (is just used to uniquely identify the model weights file)
    save_path = os.path.join(FAG_PROJEKT_DIR, "models", "model_weights", f"UnrotTrain_{model.name}_model_weights{model_specifications}.pth")
    
    model_state = {
        "model_name": model.name,
        "model_args": model_args[model],
        "state_dict": model.state_dict(),
    }
    torch.save(model_state, save_path)

# PLOT ========================================================================================
colors = {"CNN": "tab:blue", "GE_CNN": "tab:orange"}
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

for name, ((tr_loss, tr_acc), (te_loss, te_acc)) in histories.items():
    epochs = range(1, N_EPOCHS + 1)
    ax_loss.plot(epochs, tr_loss, color=colors[name], label=f"{name} train (unrotated)")
    ax_loss.plot(epochs, te_loss, color=colors[name], linestyle="--", label=f"{name} test (rotated)")
    ax_acc.plot(epochs, tr_acc, color=colors[name], label=f"{name} train (unrotated)")
    ax_acc.plot(epochs, te_acc, color=colors[name], linestyle="--", label=f"{name} test (rotated)")

ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-entropy loss")
ax_loss.set_title("Loss"); ax_loss.legend()
ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy"); ax_acc.set_ylim(0, 1); ax_acc.legend()

fig.suptitle("Train on unrotated, test on rotated: CNN vs GE_CNN")
fig.tight_layout()
fig.savefig(FAG_PROJEKT_DIR / "reports" / "cnn_vs_gecnn_unrot_train_rot_test.png", dpi=150)
plt.show()






# BASE_DIR = Path(__file__).resolve().parent
# path = os.path.join(BASE_DIR, "..", "models", "model_weights", "model_weights.pth")
# os.makedirs(os.path.dirname(path), exist_ok=True)
# print(__file__)
# print(Path(__file__))
# print(os.path.exists(path))
# torch.save(model2.state_dict(), path)