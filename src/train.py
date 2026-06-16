import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

# Change import path to a level up (from src -> Fagprojekt) so 'models' can be imported
FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from models.Model import *
from src.data import RotatedMNIST


def train_loop(model: nn.Module, lr: float, train_loader: DataLoader, n_epochs: int, test_loader: DataLoader | None = None, show_progress:bool = True, device: str | torch.device | None = None) -> tuple[tuple[list, list], tuple[list, list]]:
    """Train the model and return the average loss per epoch. alpha is the learning rate.

    If test_loader is given, the model is evaluated on it after every epoch.
    device: None -> auto CUDA, falling back to CPU.
    """
    #print('Training model: ', model)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    test_lossess = [] if test_loader is not None else None
    test_accuracy = [] if test_loader is not None else None
    for epoch in tqdm(range(n_epochs), desc = f'% Done'):
        model.train()
        total_loss = 0.0
        n_seen = 0
        n_correct = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}", disable= not show_progress)
        for images, labels, angles in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()
            n_seen += images.size(0)
            progress.set_postfix(avg_loss=f"{total_loss / n_seen:.4f}")

        train_accuracies.append(n_correct/n_seen)
        train_losses.append(total_loss / n_seen)

        if (test_loader is not None):
            test_loss, test_acc = evaluate(model, test_loader, show_progress= show_progress, device=device)
            test_lossess.append(test_loss)
            test_accuracy.append(test_acc)
            if show_progress:
                print(f"Epoch {epoch + 1}/{n_epochs} - test loss: {test_loss:.4f} - test accuracy: {test_acc:.2%}")
    print(f'Model finished training, test loss is {test_lossess[-1]}, test acc: {test_accuracy[-1]}')
    
    return (train_losses, train_accuracies), (test_lossess, test_accuracy)


def evaluate(model: nn.Module, dataloader: DataLoader, show_progress:bool = True, device: str | torch.device | None = None) -> tuple[float, float]:
    """Evaluate the model and return (average loss, accuracy)."""
    criterion = nn.CrossEntropyLoss()
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.to(device)

    model.to(device)
    model.eval()
    total_loss = 0.0
    n_correct = 0

    with torch.no_grad():
        for images, labels, angles in tqdm(dataloader, desc="Evaluating", disable= not show_progress):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()

    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, n_correct / n_samples


if __name__ == "__main__":  
    
    #os.chdir("Fagprojekt---Group-Equivariance")
    # Fraction of the dataset used for training/evaluation (1.0 = everything).
    TRAIN_FRACTION = 0.01
    TEST_FRACTION = 0.01

    MODEL = "GE_CNN" #choose CNN or GE_CNN
    train_dataset = RotatedMNIST(split="train", rotated=True,  fraction=TRAIN_FRACTION)
    test_dataset = RotatedMNIST(split="test", rotated= True, fraction=TEST_FRACTION)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # n_conv_layers=3 does not work with kernel_size=5 on 28x28: the feature maps
    # shrink to 3x3 before the last block (convs have no padding), too small for a 5x5 kernel.
    
    if MODEL == "CNN":
        model_args = {
            "kernel_size": 5,
            "in_features": 1,
            "img_size": 28,
            "n_conv_layers": 2,
            "conv_pr_pool": 1,
            "channels": 8,
            "n_classes": 10,
            "bias": True,
        }
        CNN_model = CNN(**model_args)
        CNN_model.name = "CNN"
        current_model = CNN_model
    elif MODEL == "GE_CNN":
        model_args = {
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
        GE_CNN_model = GE_CNN(**model_args)
        GE_CNN_model.name = "GE_CNN"
        current_model = GE_CNN_model
    

    
    train_loop(current_model, lr=1e-3, train_loader=train_loader, n_epochs=2, test_loader=test_loader)
    
    #SAVE the model weights
    BASE_DIR = Path(__file__).resolve().parents[1] # home directory (Fagprojekt---GE)
    model_specifications = "_" # string indicating parameters of model (is just used to uniquely identify the model weights file)
    save_path = os.path.join(BASE_DIR, "models", "model_weights", f"{current_model.name}_model_weights{model_specifications}.pth")
    
    model_state = {
        "model_name": current_model.name,
        "model_args": model_args,
        "state_dict": current_model.state_dict(),
    }

    torch.save(model_state, save_path)