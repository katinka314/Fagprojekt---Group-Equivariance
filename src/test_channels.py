
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
L           = 2
KERNEL_SIZE = 5
N_EPOCHS    = 5
BATCH_SIZE  = 128
LR          = 1e-3
IMG_SIZE    = 28
N_CLASSES   = 10

CHANNEL_CONFIGS   = [8, 16, 32, 64]   # kanalantal vi tester, har bare valgt nogle forskellige...
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

    # Dummy forward så LazyLinear initialiseres inden parameteroptælling
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    ge(dummy)
    cnn(dummy)

    ge_params  = sum(p.numel() for p in ge.parameters())
    cnn_params = sum(p.numel() for p in cnn.parameters())

    print(f"\n--- channels={channels} ---")
    print(f"GE-CNN  parametre: {ge_params:>10,}")
    print(f"CNN     parametre: {cnn_params:>10,}")

    ge_history  = train_loop(ge,  lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS, test_loader=test_loader, show_progress=False)
    cnn_history = train_loop(cnn, lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS, test_loader=test_loader, show_progress=False)

    ge_loss,  ge_acc  = ge_history[1][0][-1],  ge_history[1][1][-1]
    cnn_loss, cnn_acc = cnn_history[1][0][-1], cnn_history[1][1][-1]

    results.append({
        "channels":         channels,
        "ge_params":        ge_params,
        "cnn_params":       cnn_params,
        "ge_acc":           ge_acc,
        "cnn_acc":          cnn_acc,
        "ge_loss":          ge_loss,
        "cnn_loss":         cnn_loss,
        "ge_train_losses":  ge_history[0][0],
        "ge_train_accs":    ge_history[0][1],
        "ge_test_losses":   ge_history[1][0],
        "ge_test_accs":     ge_history[1][1],
        "cnn_train_losses": cnn_history[0][0],
        "cnn_train_accs":   cnn_history[0][1],
        "cnn_test_losses":  cnn_history[1][0],
        "cnn_test_accs":    cnn_history[1][1],
    })


# plot Resultater

print("\n=== Test 1: Samme kanalantal ===")
print(f"{'channels':>10} {'GE params':>12} {'CNN params':>12} {'GE acc':>10} {'CNN acc':>10}")
print("-" * 58)
for r in results:
    print(f"{r['channels']:>10} {r['ge_params']:>12,} {r['cnn_params']:>12,} "
          f"{r['ge_acc']:>10.2%} {r['cnn_acc']:>10.2%}")

# Gem resultater til disk
RESULTS_PATH = FAG_PROJEKT_DIR / "results" / "test_channels_results.pt"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(results, RESULTS_PATH)
print(f"\nResultater gemt til {RESULTS_PATH}")

