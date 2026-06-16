"""Train on UNROTATED MNIST, test on ROTATED MNIST: CNN vs GE-CNN.
    CNN     channels=16  ~ 95,882 params
    GE-CNN  channels=8   ~ 69,570 params 
"""

import json
import os
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # headless: HPC compute nodes have no display
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from data import RotatedMNIST
from train import train_loop, evaluate
from models.Model import CNN, GE_CNN


# Configuration  (model sizes locked: GE-CNN channels=8, CNN channels=16 -> ~equal params)
def _env_float(name, default):
    return float(os.environ.get(name, default))

def _env_int(name, default):
    return int(os.environ.get(name, default))

L              = 2
KERNEL_SIZE    = 5
BATCH_SIZE     = _env_int("BATCH_SIZE", 128)
LR             = _env_float("LR", 1e-3)
IMG_SIZE       = 28
N_CLASSES      = 10
N_RINGS        = 4
N_CONV_LAYERS  = 2
CONV_PR_POOL   = 1
TRAIN_FRACTION = _env_float("TRAIN_FRACTION", 0.1)
TEST_FRACTION  = _env_float("TEST_FRACTION", 1.0)
# Many more epochs than before (was 10) so both models train close to convergence.
N_EPOCHS       = _env_int("N_EPOCHS", 100)
SEED           = _env_int("SEED", 0)

DEVICE = os.environ.get("DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")

OUT_DIR = FAG_PROJEKT_DIR / "reports" / "unrot_rot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(SEED)

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Epochs: {N_EPOCHS} | batch: {BATCH_SIZE} | lr: {LR} | "
      f"train_frac: {TRAIN_FRACTION} | test_frac: {TEST_FRACTION}")


# ----------------------------------------------------------------------------------------
# Data:  train on UNROTATED, test on ROTATED (main) and on UNROTATED (in-distribution ref)
# ----------------------------------------------------------------------------------------
train_dataset    = RotatedMNIST(split="train", rotated=False, fraction=TRAIN_FRACTION)
test_rot_dataset = RotatedMNIST(split="test",  rotated=True,  fraction=TEST_FRACTION)
test_unr_dataset = RotatedMNIST(split="test",  rotated=False, fraction=TEST_FRACTION)

train_loader    = DataLoader(train_dataset,    batch_size=BATCH_SIZE, shuffle=True)
test_rot_loader = DataLoader(test_rot_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_unr_loader = DataLoader(test_unr_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ----------------------------------------------------------------------------------------
# Models  (parameter-matched)
# ----------------------------------------------------------------------------------------
ge_args = dict(kernel_size=KERNEL_SIZE, l=L, in_features=1, img_size=IMG_SIZE,
               n_conv_layers=N_CONV_LAYERS, conv_pr_pool=CONV_PR_POOL, channels=8,
               n_classes=N_CLASSES, bias=True, n_rings=N_RINGS)
cnn_args = dict(kernel_size=KERNEL_SIZE, in_features=1, img_size=IMG_SIZE,
                n_conv_layers=N_CONV_LAYERS, conv_pr_pool=CONV_PR_POOL, channels=16,
                n_classes=N_CLASSES, bias=True)

GE_CNN_model = GE_CNN(**ge_args); GE_CNN_model.name = "GE_CNN"
CNN_model    = CNN(**cnn_args);   CNN_model.name = "CNN"

# Dummy forward initialises the LazyLinear layers (on CPU) so parameter counts are exact;
# train_loop then moves each model to DEVICE.
dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)
for m in (GE_CNN_model, CNN_model):
    m(dummy)

models = {"CNN": CNN_model, "GE_CNN": GE_CNN_model}
model_args = {"CNN": cnn_args, "GE_CNN": ge_args}
n_params = {name: sum(p.numel() for p in m.parameters()) for name, m in models.items()}


# ----------------------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------------------
histories = {}     # name -> ((train_loss, train_acc), (test_rot_loss, test_rot_acc))
final = {}         # name -> dict of final metrics
for name, m in models.items():
    print(f"\n=== Training {name} ({n_params[name]:,} parameters) ===")
    hist = train_loop(m, lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS,
                      test_loader=test_rot_loader, show_progress=False, device=DEVICE)
    histories[name] = hist
    # Final in-distribution (unrotated) test accuracy for reference.
    unr_loss, unr_acc = evaluate(m, test_unr_loader, show_progress=False, device=DEVICE)
    (tr_loss, tr_acc), (rot_loss, rot_acc) = hist
    final[name] = {
        "n_params": n_params[name],
        "train_loss": tr_loss[-1], "train_acc": tr_acc[-1],
        "test_rot_loss": rot_loss[-1], "test_rot_acc": rot_acc[-1],
        "test_unrot_loss": unr_loss, "test_unrot_acc": unr_acc,
    }
    print(f"{name}: train acc {tr_acc[-1]:.2%} | test(unrot) {unr_acc:.2%} | "
          f"test(rot) {rot_acc[-1]:.2%}")


# ----------------------------------------------------------------------------------------
# Save raw data + summary
# ----------------------------------------------------------------------------------------
config = dict(L=L, kernel_size=KERNEL_SIZE, batch_size=BATCH_SIZE, lr=LR, img_size=IMG_SIZE,
              n_classes=N_CLASSES, n_rings=N_RINGS, n_conv_layers=N_CONV_LAYERS,
              conv_pr_pool=CONV_PR_POOL, train_fraction=TRAIN_FRACTION,
              test_fraction=TEST_FRACTION, n_epochs=N_EPOCHS, seed=SEED, device=DEVICE)

torch.save({"config": config, "histories": histories, "final": final,
            "n_params": n_params, "model_args": model_args},
           OUT_DIR / "histories.pt")

with open(OUT_DIR / "summary.json", "w") as f:
    json.dump({"config": config, "n_params": n_params, "final": final}, f, indent=2)

# Model weights, in the same format train.py uses (name + args + state_dict) so they reload.
for name, m in models.items():
    torch.save({"model_name": name, "model_args": model_args[name],
                "state_dict": m.state_dict()},
               OUT_DIR / f"{name}_model_weights.pth")


# ----------------------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------------------
colors = {"CNN": "tab:blue", "GE_CNN": "tab:orange"}
epochs = range(1, N_EPOCHS + 1)

# 1) Learning curves: loss and accuracy (train-unrotated vs test-rotated) over epochs.
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
for name, ((tr_loss, tr_acc), (rot_loss, rot_acc)) in histories.items():
    lbl = f"{name} ({n_params[name]:,}p)"
    ax_loss.plot(epochs, tr_loss, color=colors[name], label=f"{lbl} train (unrot)")
    ax_loss.plot(epochs, rot_loss, color=colors[name], linestyle="--", label=f"{lbl} test (rot)")
    ax_acc.plot(epochs, tr_acc, color=colors[name], label=f"{lbl} train (unrot)")
    ax_acc.plot(epochs, rot_acc, color=colors[name], linestyle="--", label=f"{lbl} test (rot)")
ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-entropy loss")
ax_loss.set_title("Loss"); ax_loss.legend(fontsize=8)
ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy"); ax_acc.set_ylim(0, 1); ax_acc.legend(fontsize=8)
fig.suptitle("Train on unrotated, test on rotated: CNN vs GE-CNN (parameter-matched)")
fig.tight_layout()
fig.savefig(OUT_DIR / "learning_curves.png", dpi=150)
plt.close(fig)

# 2) Final accuracy bars: in-distribution (unrotated) vs rotated, per model.
fig, ax = plt.subplots(figsize=(7, 5))
labels = list(models.keys())
x = range(len(labels))
unrot = [final[n]["test_unrot_acc"] for n in labels]
rot   = [final[n]["test_rot_acc"] for n in labels]
w = 0.35
ax.bar([i - w/2 for i in x], unrot, w, label="test unrotated (in-distribution)", color="tab:green")
ax.bar([i + w/2 for i in x], rot,   w, label="test rotated (out-of-distribution)", color="tab:red")
for i, (u, r) in enumerate(zip(unrot, rot)):
    ax.text(i - w/2, u + 0.01, f"{u:.0%}", ha="center", fontsize=9)
    ax.text(i + w/2, r + 0.01, f"{r:.0%}", ha="center", fontsize=9)
ax.set_xticks(list(x))
ax.set_xticklabels([f"{n}\n({n_params[n]:,}p)" for n in labels])
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.set_title("Final test accuracy: unrotated vs rotated")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "final_accuracy_bars.png", dpi=150)
plt.close(fig)

# 3) Robustness gap: how much accuracy is lost going unrotated -> rotated (lower = better).
fig, ax = plt.subplots(figsize=(7, 5))
gaps = [final[n]["test_unrot_acc"] - final[n]["test_rot_acc"] for n in labels]
bars = ax.bar(labels, gaps, color=[colors[n] for n in labels])
for b, g in zip(bars, gaps):
    ax.text(b.get_x() + b.get_width()/2, g + 0.005, f"{g:.1%}", ha="center", fontsize=10)
ax.set_ylabel("Accuracy drop (unrotated - rotated)")
ax.set_title("Rotation-robustness gap (lower = more equivariant)")
fig.tight_layout()
fig.savefig(OUT_DIR / "robustness_gap.png", dpi=150)
plt.close(fig)


print(f"\nAll outputs written to {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")
