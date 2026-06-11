
#test 1for kanaler

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from models.Model import GE_CNN, CNN
from src.data import RotatedMNIST
from src.train import train_loop, evaluate


#hyperparametre inits
L           = 1
KERNEL_SIZE = 5
N_EPOCHS    = 5
BATCH_SIZE  = 128
LR          = 1e-3
IMG_SIZE    = 28
N_CLASSES   = 10

CHANNEL_CONFIGS   = [4, 8, 16, 32]   # kanalantal vi tester, har bare valgt nogle forskellige...
TRAIN_FRACTION    = 0.1 #bare for mindre køretid ligenu
TEST_FRACTION     = 1.0

# Datafis

train_dataset = RotatedMNIST(split="train", rotated=True,  fraction=TRAIN_FRACTION)
test_dataset  = RotatedMNIST(split="test",  rotated=True,  fraction=TEST_FRACTION)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


# Test 1 hvor vi sørger for at de to modeller har samme kanalantal

results = []
#hoved loop hvor vi ikke ganger GE-CNN med len_basis fordi det giver unfair sammenglinging
for channels in CHANNEL_CONFIGS:

    ge  = GE_CNN(kernel_size=KERNEL_SIZE, l=L, channels=channels,
                 img_size=IMG_SIZE, n_classes=N_CLASSES, n_conv_layers=2, conv_pr_pool=1)
    cnn = CNN(kernel_size=KERNEL_SIZE, channels=channels,
                 img_size=IMG_SIZE, n_classes=N_CLASSES, n_conv_layers=2, conv_pr_pool=1)

    ge_params  = sum(p.numel() for p in ge.parameters())
    cnn_params = sum(p.numel() for p in cnn.parameters())

    print(f"\n--- channels={channels} ---")
    print(f"GE-CNN  parametre: {ge_params:>10,}")
    print(f"CNN     parametre: {cnn_params:>10,}")

    train_loop(ge,  lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS, test_loader=test_loader)
    train_loop(cnn, lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS, test_loader=test_loader)

    # tilføjet så vi kan se hvad parameterforskellen faktisk er på de forskellige antal kanaler.
    ge_loss,  ge_acc  = evaluate(ge,  test_loader)
    cnn_loss, cnn_acc = evaluate(cnn, test_loader)

    results.append({
        "channels":    channels,
        "ge_params":   ge_params,
        "cnn_params":  cnn_params,
        "ge_acc":      ge_acc,
        "cnn_acc":     cnn_acc,
        "ge_loss":     ge_loss,
        "cnn_loss":    cnn_loss,
    })


# plot Resultater

print("\n=== Test 1: Samme kanalantal ===")
print(f"{'channels':>10} {'GE params':>12} {'CNN params':>12} {'GE acc':>10} {'CNN acc':>10}")
print("-" * 58)
for r in results:
    print(f"{r['channels']:>10} {r['ge_params']:>12,} {r['cnn_params']:>12,} "
          f"{r['ge_acc']:>10.2%} {r['cnn_acc']:>10.2%}")

