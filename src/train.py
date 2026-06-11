import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

print("current dir:", os.getcwd())
# Add fag_projekt/ to sys.path so 'models' can be imported
FAG_PROJEKT_DIR = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(FAG_PROJEKT_DIR))

print("current dir:", os.getcwd())
from models.Model import *
from src.data import RotatedMNIST


def train_loop(model: nn.Module, lr: float, dataloader: DataLoader, n_epochs: int, test_loader: DataLoader | None = None) -> list[float]:
    """Train the model and return the average loss per epoch. alpha is the learning rate.

    If test_loader is given, the model is evaluated on it after every epoch.
    """
    #print('Training model: ', model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    epoch_losses = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for images, labels, angles in progress:
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
            progress.set_postfix(avg_loss=f"{total_loss / n_seen:.4f}")

        epoch_losses.append(total_loss / n_seen)

        if test_loader is not None:
            test_loss, test_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch + 1}/{n_epochs} - test loss: {test_loss:.4f} - test accuracy: {test_acc:.2%}")

    return epoch_losses


def evaluate(model: nn.Module, dataloader: DataLoader) -> tuple[float, float]:
    """Evaluate the model and return (average loss, accuracy)."""
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    n_correct = 0

    with torch.no_grad():
        for images, labels, angles in tqdm(dataloader, desc="Evaluating"):
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            n_correct += (logits.argmax(dim=1) == labels).sum().item()

    n_samples = len(dataloader.dataset)
    return total_loss / n_samples, n_correct / n_samples


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    print(os.getcwd())
    # Fraction of the dataset used for training/evaluation (1.0 = everything).
    TRAIN_FRACTION = 0.1
    TEST_FRACTION = 0.1

    train_dataset = RotatedMNIST(split="train", fraction=TRAIN_FRACTION)
    test_dataset = RotatedMNIST(split="test", fraction=TEST_FRACTION)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # n_conv_layers=3 does not work with kernel_size=5 on 28x28: the feature maps
    # shrink to 3x3 before the last block (convs have no padding), too small for a 5x5 kernel.
    model = GE_CNN(
        kernel_size=5,
        l=1,
        in_features=1,
        img_size=28,
        n_conv_layers=2,
        conv_pr_pool=1,
        channels=8,
        n_classes=10,
        bias=True,
    )
    
    model2 = CNN(kernel_size= 5, in_features = 1, img_size = 28, n_conv_layers =2, conv_pr_pool = 1, channels = 8, n_classes= 10, bias = True)

    train_loop(model2, alpha=1e-3, dataloader=train_loader, n_epochs=2, test_loader=test_loader)
    
    path = os.path.join(BASE_DIR,"..", "models", "model_weights", "model_weights.pth")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(__file__)
    print(Path(__file__))
    print(os.path.exists(path))
    torch.save(model2.state_dict(), path)