
# Test 2: sammenlign GE-CNN og CNN med samme antal parametre

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
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
TRAIN_FRACTION  = 0.1
TEST_FRACTION   = 1.0

# ideen er at vi har et antal faste CNN channel som vi tester. og for hver værdi i CNN_CHANNEL_CONFIG,
# så finder vi et GE-CNN med det antal parametre der tættest matcher den af det normale CNN ved det valgte,
# antal channels

# Faste CNN kanalstørrelser vi tester — GE-CNN matches til disse
CNN_CHANNEL_CONFIGS = [2, 4, 8, 16, 32, 64, 128]

ARCH_KWARGS = dict(
    kernel_size   = KERNEL_SIZE,
    img_size      = IMG_SIZE,
    n_classes     = N_CLASSES,
    n_conv_layers = 2,
    conv_pr_pool  = 1,
)

# n_rings er IKKE fast længere — find_matching_ge_config søger over den for at
# ramme CNN'ens parametertal, så GE-CNN'ens størrelse kan finjusteres.
GE_KWARGS  = dict(**ARCH_KWARGS)
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


def find_matching_ge_config(target_params,
                            channel_opts=(2, 4, 8, 16, 32, 64, 128),
                            nring_range=range(2, KERNEL_SIZE + 1)):
    """Find (channels, n_rings) der giver tættest match på target_params.
    channels snappes alligevel til toer-potens i GE_CNN, så vi søger kun dem,
    og finjusterer parametertallet med n_rings.

    n_rings cappes ved KERNEL_SIZE: en kxk-kerne har kun ~k distinkte radier
    (for 5x5: radierne {0,1,2,4,5,8} -> 6 stk), så flere ringe end det lægger
    blot overlappende Gaussians på de samme pixels -> redundante parametre."""
    best = None  # (diff, channels, n_rings, n_params)
    for ch in channel_opts:
        for nr in nring_range:
            ge = GE_CNN(l=L, channels=ch, n_rings=nr, **GE_KWARGS)
            n = count_params(ge)
            diff = abs(n - target_params)
            if best is None or diff < best[0]:
                best = (diff, ch, nr, n)
    return best  # diff, channels, n_rings, n_params


# Hoved loop
results = []

for cnn_channels in CNN_CHANNEL_CONFIGS:
    cnn = CNN(channels=cnn_channels, **CNN_KWARGS)
    cnn_params = count_params(cnn)

    param_diff, matched_ge_channels, matched_n_rings, ge_params = find_matching_ge_config(cnn_params)
    ge = GE_CNN(l=L, channels=matched_ge_channels, n_rings=matched_n_rings, **GE_KWARGS)

    print(f"\n--- CNN channels={cnn_channels} ---")
    print(f"CNN parametre:            {cnn_params:>10,}")
    print(f"GE-CNN matched channels:  {matched_ge_channels:>10}")
    print(f"GE-CNN matched n_rings:   {matched_n_rings:>10}")
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
        "ge_n_rings":       matched_n_rings,
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
print(f"{'CNN ch':>7} {'GE ch':>6} {'GE rings':>9} {'CNN params':>12} {'GE params':>12} {'forskel':>10} {'GE acc':>10} {'CNN acc':>10}")
print("-" * 82)
for r in results:
    print(f"{r['cnn_channels']:>7} {r['ge_channels']:>6} {r['ge_n_rings']:>9} {r['cnn_params']:>12,} {r['ge_params']:>12,} "
          f"{r['param_diff']:>10,} {r['ge_acc']:>10.2%} {r['cnn_acc']:>10.2%}")

# Gem
RESULTS_PATH = FAG_PROJEKT_DIR / "results" / "test_parameters_results.pt"
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(results, RESULTS_PATH)
print(f"\nResultater gemt til {RESULTS_PATH}")


# Visualisering
def plot_results(results, save_dir=FAG_PROJEKT_DIR / "reports"):
    """Tegn (1) accuracy vs. parametertal for begge modeller, og
    (2) test-accuracy pr. epoke for hver matchet parameter-config."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rs = sorted(results, key=lambda r: r["cnn_params"])
    cnn_params = [r["cnn_params"] for r in rs]
    ge_params  = [r["ge_params"]  for r in rs]
    cnn_accs   = [r["cnn_acc"]    for r in rs]
    ge_accs    = [r["ge_acc"]     for r in rs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    #Accuracy vs. parametertal hver model plottes mod sit EGET parametertal,
    # så den lille matchforskel ikke skjules.
    ax1.plot(ge_params,  ge_accs,  "o-", label="GE-CNN", color="tab:blue")
    ax1.plot(cnn_params, cnn_accs, "s-", label="CNN",    color="tab:orange")
    ax1.set_xscale("log")
    ax1.set_xlabel("Antal parametre (log-skala)")
    ax1.set_ylabel("Test-accuracy")
    ax1.set_title("Accuracy vs. parametertal")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    #Test-accuracy pr. epoke for hver config.
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(rs):
        color = cmap(i / max(len(rs) - 1, 1))
        epochs = range(1, len(r["ge_test_accs"]) + 1)
        label = f"~{r['cnn_params']/1000:.0f}k par"
        ax2.plot(epochs, r["ge_test_accs"],  "o-", color=color, label=f"GE {label}")
        ax2.plot(epochs, r["cnn_test_accs"], "s--", color=color, label=f"CNN {label}")
    ax2.set_xlabel("Epoke")
    ax2.set_ylabel("Test-accuracy")
    ax2.set_title("Test-accuracy pr. epoke (heltrukket=GE, stiplet=CNN)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, ncol=2)

    fig.suptitle("Test 2: GE-CNN vs CNN ved matchet parameterantal", fontsize=13)
    fig.tight_layout()

    fig_path = save_dir / "test_parameters_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Figur gemt til {fig_path}")
    plt.show()


plot_results(results)
