
# Test 2: sammenlign GE-CNN og CNN med samme antal parametre

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from models.Model import GE_CNN, CNN
from src.data import RotatedMNIST
from src.train import train_loop


# Hyperparametre
L               = 2
KERNEL_SIZE     = 5
N_EPOCHS        = 5
BATCH_SIZE      = 128
LR              = 1e-3
IMG_SIZE        = 28
N_CLASSES       = 10
N_RINGS         = 4
TRAIN_FRACTION  = 0.1
TEST_FRACTION   = 1.0

# ideen er at vi har et antal faste CNN channel som vi tester. og for hver værdi i CNN_CHANNEL_CONFIG,
# så finder vi et GE-CNN med det antal parametre der tættest matcher den af det normale CNN ved det valgte,
# antal channels

# Faste CNN kanalstørrelser vi tester — GE-CNN matches til disse
CNN_CHANNEL_CONFIGS = [8, 16, 32, 64]

ARCH_KWARGS = dict(
    kernel_size   = KERNEL_SIZE,
    img_size      = IMG_SIZE,
    n_classes     = N_CLASSES,
    n_conv_layers = 2,
    conv_pr_pool  = 1,
)

GE_KWARGS  = dict(**ARCH_KWARGS, n_rings=N_RINGS)
CNN_KWARGS = dict(**ARCH_KWARGS)

# Data
train_dataset = RotatedMNIST(split="train", rotated=True, fraction=TRAIN_FRACTION)
test_dataset  = RotatedMNIST(split="test",  rotated=True, fraction=TEST_FRACTION)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)


def count_params(model):
    """Kør dummy forward så LazyLinear initialiseres, tæl derefter parametre."""
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
    model(dummy)
    return sum(p.numel() for p in model.parameters())


def find_matching_ge_channels(target_params, search_range=range(1, 128)):
    """Find det GE-CNN kanalantal der giver tættest match på target_params."""
    best_channels, best_diff = None, float("inf")
    for ch in search_range:
        ge = GE_CNN(l=L, channels=ch, **GE_KWARGS)
        n = count_params(ge)
        diff = abs(n - target_params)
        if diff < best_diff:
            best_diff, best_channels = diff, ch
    return best_channels, best_diff


# Hoved loop
results = []

for cnn_channels in CNN_CHANNEL_CONFIGS:
    cnn = CNN(channels=cnn_channels, **CNN_KWARGS)
    cnn_params = count_params(cnn)

    matched_ge_channels, param_diff = find_matching_ge_channels(cnn_params)
    ge = GE_CNN(l=L, channels=matched_ge_channels, **GE_KWARGS)
    ge_params = count_params(ge)

    print(f"\n--- CNN channels={cnn_channels} ---")
    print(f"CNN parametre:            {cnn_params:>10,}")
    print(f"GE-CNN matched channels:  {matched_ge_channels:>10}")
    print(f"GE-CNN parametre:         {ge_params:>10,}  (forskel: {param_diff:,})")

    ge_history  = train_loop(ge,  lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS, test_loader=test_loader, show_progress=False)
    cnn_history = train_loop(cnn, lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS, test_loader=test_loader, show_progress=False)

    ge_acc  = ge_history[1][1][-1]
    cnn_acc = cnn_history[1][1][-1]
    ge_loss = ge_history[1][0][-1]
    cnn_loss = cnn_history[1][0][-1]

    results.append({
        "cnn_channels":     cnn_channels,
        "ge_channels":      matched_ge_channels,
        "cnn_params":       cnn_params,
        "ge_params":        ge_params,
        "param_diff":       param_diff,
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


# Resultat tabel
print("\n=== Test 2: Samme parameterantal ===")
print(f"{'CNN ch':>7} {'GE ch':>6} {'CNN params':>12} {'GE params':>12} {'forskel':>10} {'GE acc':>10} {'CNN acc':>10}")
print("-" * 72)
for r in results:
    print(f"{r['cnn_channels']:>7} {r['ge_channels']:>6} {r['cnn_params']:>12,} {r['ge_params']:>12,} "
          f"{r['param_diff']:>10,} {r['ge_acc']:>10.2%} {r['cnn_acc']:>10.2%}")

# Gem
RESULTS_PATH = FAG_PROJEKT_DIR / "results" / "test_parameters_results.pt"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(results, RESULTS_PATH)
print(f"\nResultater gemt til {RESULTS_PATH}")
