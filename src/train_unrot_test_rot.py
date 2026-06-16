"""Train on UNROTATED MNIST, test on ROTATED MNIST: CNN vs GE-CNN, over multiple seeds.
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: HPC compute nodes have no display
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from data import RotatedMNIST
from train import train_loop, evaluate
from models.Model import CNN, GE_CNN


# Config

SMOKE = True   # True = fast local smoke test, False = full overnight/HPC run
PADDING = True
if SMOKE:
    SEEDS          = range(2)
    N_EPOCHS       = 5
    TRAIN_FRACTION = 0.01
    TEST_FRACTION  = 0.05
else:
    SEEDS          = range(10)
    N_EPOCHS       = 60
    TRAIN_FRACTION = 0.1
    TEST_FRACTION  = 1.0

L              = 2
KERNEL_SIZE    = 5
BATCH_SIZE     = 128
LR             = 1e-3
N_CLASSES      = 10
N_RINGS        = 4
N_CONV_LAYERS  = 2
CONV_PR_POOL   = 1

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = FAG_PROJEKT_DIR / "reports" / "unrot_rot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAMES = ["CNN", "GE_CNN"]
COLORS = {"CNN": "tab:blue", "GE_CNN": "tab:orange"}
# Data (created once: same subset across seeds; only init + shuffle vary per seed)

train_dataset    = RotatedMNIST(split="train", rotated=False, fraction=TRAIN_FRACTION, padding=PADDING)
test_rot_dataset = RotatedMNIST(split="test",  rotated=True,  fraction=TEST_FRACTION, padding=PADDING)
test_unr_dataset = RotatedMNIST(split="test",  rotated=False, fraction=TEST_FRACTION, padding=PADDING)

IMG_SIZE = train_dataset.images.shape[-1]

ge_args = dict(kernel_size=KERNEL_SIZE, l=L, in_features=1, img_size=IMG_SIZE,
               n_conv_layers=N_CONV_LAYERS, conv_pr_pool=CONV_PR_POOL, channels=8,
               n_classes=N_CLASSES, bias=True, n_rings=N_RINGS)
cnn_args = dict(kernel_size=KERNEL_SIZE, in_features=1, img_size=IMG_SIZE,
                n_conv_layers=N_CONV_LAYERS, conv_pr_pool=CONV_PR_POOL, channels=16,
                n_classes=N_CLASSES, bias=True)
MODEL_ARGS = {"CNN": cnn_args, "GE_CNN": ge_args}



def build_models():
    models = {"CNN": CNN(**cnn_args), "GE_CNN": GE_CNN(**ge_args)}
    for name, m in models.items():
        m.name = name
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)   # materialise LazyLinear on CPU
    for m in models.values():
        m(dummy)
    return models


def run_one_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    g = torch.Generator().manual_seed(seed)
    train_loader    = DataLoader(train_dataset,    batch_size=BATCH_SIZE, shuffle=True, generator=g)
    test_rot_loader = DataLoader(test_rot_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_unr_loader = DataLoader(test_unr_dataset, batch_size=BATCH_SIZE, shuffle=False)

    models = build_models()
    n_params = {n: sum(p.numel() for p in m.parameters()) for n, m in models.items()}

    histories, final = {}, {}
    for name, m in models.items():
        print(f"  [seed {seed}] training {name} ({n_params[name]:,} params)")
        hist = train_loop(m, lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS,
                          test_loader=test_rot_loader, show_progress=False, device=DEVICE)
        (tr_loss, tr_acc), (rot_loss, rot_acc) = hist
        unr_loss, unr_acc = evaluate(m, test_unr_loader, show_progress=False, device=DEVICE)
        histories[name] = hist
        final[name] = {
            "train_loss": tr_loss[-1], "train_acc": tr_acc[-1],
            "test_rot_loss": rot_loss[-1], "test_rot_acc": rot_acc[-1],
            "test_unrot_loss": unr_loss, "test_unrot_acc": unr_acc,
        }
    return histories, final, n_params


def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    return mean, std


# Run all seeds

print(f"Device: {DEVICE}, seeds={list(SEEDS)}, epochs={N_EPOCHS}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

all_histories, all_final = [], []
n_params = None
for seed in SEEDS:
    print(f"\nSEED {seed}")
    histories, final, n_params = run_one_seed(seed)
    all_histories.append(histories)
    all_final.append(final)

n_seeds = len(SEEDS)


# Aggregate

# curves[name][key] -> array (n_seeds, n_epochs)
curves = {}
for name in MODEL_NAMES:
    curves[name] = {
        "train_loss":    np.array([h[name][0][0] for h in all_histories]),
        "train_acc":     np.array([h[name][0][1] for h in all_histories]),
        "test_rot_loss": np.array([h[name][1][0] for h in all_histories]),
        "test_rot_acc":  np.array([h[name][1][1] for h in all_histories]),
    }

FINAL_KEYS = ["train_acc", "test_rot_acc", "test_unrot_acc"]
final_stats = {}
for name in MODEL_NAMES:
    final_stats[name] = {}
    for k in FINAL_KEYS:
        vals = np.array([f[name][k] for f in all_final], dtype=float)
        m, s = mean_std(vals)
        final_stats[name][k] = {"mean": float(m), "std": float(s), "values": vals.tolist()}

# Robustness gap per seed, then mean/std
gap_vals = {n: np.array([f[n]["test_unrot_acc"] - f[n]["test_rot_acc"] for f in all_final])
            for n in MODEL_NAMES}

print("\nAggregate (mean +/- std over seeds)")
for name in MODEL_NAMES:
    for k in FINAL_KEYS:
        st = final_stats[name][k]
        print(f"  {name:7s} {k:16s} {st['mean']:.4f} +/- {st['std']:.4f}")


# Save

config = dict(L=L, kernel_size=KERNEL_SIZE, batch_size=BATCH_SIZE, lr=LR, img_size=IMG_SIZE,
              n_classes=N_CLASSES, n_rings=N_RINGS, n_conv_layers=N_CONV_LAYERS,
              conv_pr_pool=CONV_PR_POOL, train_fraction=TRAIN_FRACTION,
              test_fraction=TEST_FRACTION, n_epochs=N_EPOCHS, seeds=list(SEEDS),
              smoke=SMOKE, device=DEVICE)

torch.save({"config": config, "all_histories": all_histories, "all_final": all_final,
            "curves": curves, "n_params": n_params, "model_args": MODEL_ARGS},
           OUT_DIR / "aggregate.pt")

with open(OUT_DIR / "aggregate.json", "w") as f:
    json.dump({"config": config, "n_params": n_params, "n_seeds": n_seeds,
               "final_stats": final_stats}, f, indent=2)


# Plots (mean line + std band / error bars)

epochs = range(1, N_EPOCHS + 1)

# 1) Learning curves with std bands
fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 5))
for name in MODEL_NAMES:
    c = COLORS[name]
    lbl = f"{name} ({n_params[name]:,}p)"
    for ax, loss_key, acc_key in [(ax_loss, "train_loss", None), (ax_acc, "train_acc", None)]:
        pass
    # train (unrot)
    m_trl, s_trl = mean_std(curves[name]["train_loss"])
    m_tra, s_tra = mean_std(curves[name]["train_acc"])
    # test (rot)
    m_rol, s_rol = mean_std(curves[name]["test_rot_loss"])
    m_roa, s_roa = mean_std(curves[name]["test_rot_acc"])

    ax_loss.plot(epochs, m_trl, color=c, label=f"{lbl} train (unrot)")
    ax_loss.fill_between(epochs, m_trl - s_trl, m_trl + s_trl, color=c, alpha=0.15)
    ax_loss.plot(epochs, m_rol, color=c, linestyle="--", label=f"{lbl} test (rot)")
    ax_loss.fill_between(epochs, m_rol - s_rol, m_rol + s_rol, color=c, alpha=0.15)

    ax_acc.plot(epochs, m_tra, color=c, label=f"{lbl} train (unrot)")
    ax_acc.fill_between(epochs, m_tra - s_tra, m_tra + s_tra, color=c, alpha=0.15)
    ax_acc.plot(epochs, m_roa, color=c, linestyle="--", label=f"{lbl} test (rot)")
    ax_acc.fill_between(epochs, m_roa - s_roa, m_roa + s_roa, color=c, alpha=0.15)

ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross-entropy loss")
ax_loss.set_title("Loss"); ax_loss.legend(fontsize=8)
ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
ax_acc.set_title("Accuracy"); ax_acc.set_ylim(0, 1); ax_acc.legend(fontsize=8)
fig.suptitle(f"Train unrotated, test rotated: CNN vs GE-CNN (mean +/- std over {n_seeds} seeds)")
fig.tight_layout()
fig.savefig(OUT_DIR / "learning_curves.png", dpi=150)
plt.close(fig)

# 2) Final accuracy bars with std error bars
fig, ax = plt.subplots(figsize=(7, 5))
x = range(len(MODEL_NAMES)); w = 0.35
unrot_m = [final_stats[n]["test_unrot_acc"]["mean"] for n in MODEL_NAMES]
unrot_s = [final_stats[n]["test_unrot_acc"]["std"] for n in MODEL_NAMES]
rot_m   = [final_stats[n]["test_rot_acc"]["mean"] for n in MODEL_NAMES]
rot_s   = [final_stats[n]["test_rot_acc"]["std"] for n in MODEL_NAMES]
ax.bar([i - w/2 for i in x], unrot_m, w, yerr=unrot_s, capsize=4,
       label="test unrotated (in-distribution)", color="tab:green")
ax.bar([i + w/2 for i in x], rot_m, w, yerr=rot_s, capsize=4,
       label="test rotated (out-of-distribution)", color="tab:red")
ax.set_xticks(list(x))
ax.set_xticklabels([f"{n}\n({n_params[n]:,}p)" for n in MODEL_NAMES])
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.set_title(f"Final test accuracy (mean +/- std over {n_seeds} seeds)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "final_accuracy_bars.png", dpi=150)
plt.close(fig)

# 3) Robustness gap with std error bars
fig, ax = plt.subplots(figsize=(7, 5))
gap_m = [gap_vals[n].mean() for n in MODEL_NAMES]
gap_s = [float(mean_std(gap_vals[n])[1]) for n in MODEL_NAMES]
ax.bar(MODEL_NAMES, gap_m, yerr=gap_s, capsize=4, color=[COLORS[n] for n in MODEL_NAMES])
ax.set_ylabel("Accuracy drop (unrotated - rotated)")
ax.set_title(f"Rotation-robustness gap (mean +/- std over {n_seeds} seeds, lower = better)")
fig.tight_layout()
fig.savefig(OUT_DIR / "robustness_gap.png", dpi=150)
plt.close(fig)

print(f"\nAll outputs written to {OUT_DIR}")
for p in sorted(OUT_DIR.iterdir()):
    print(f"  {p.name}")