"""Datasize x channels grid: CNN vs GE-CNN, multi-seed, early stopping.

v3 difference from v2:
  - No parameter matching. Both models use the SAME channel widths; we compare
    them at equal channels and report each model's own parameter count.
  - Full 5 x 5 grid (datasize x channels) instead of two separate 1D sweeps.
  - One small HPC job PER CELL (one datasize x one channel value), mirroring the
    train_unrot_test_rot submit pattern. The cell is selected via env vars
    TRAIN_FRACTION and CHANNELS, so submit_param_datasize_v3.sh can fan out 25 jobs.
  - Early stopping (variable epochs): train up to MAX_EPOCHS, stop when the monitor
    loss has not improved for PATIENCE epochs, restore the best-epoch weights.
  - Saved per cell, per model, per seed: trained model weights, full per-epoch
    train/test loss+accuracy history, final accuracy, and per-class accuracy for
    BOTH train and test. Plus each model's parameter count.

Raw, reproducible per-seed results only. Plots are assembled afterwards
(see aggregate_param_datasize_v3.py).
"""

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from data import RotatedMNIST
from train import evaluate
from models.Model import CNN, GE_CNN


# Config

SMOKE = False   # True = tiny local sanity run, False = full HPC run
PADDING = True

# The cell of the grid handled by THIS job. env vars are used ONLY here, so the
# submit script can fan out one job per (datasize, channels) cell to the HPC.
TRAIN_FRACTION = float(os.environ.get("TRAIN_FRACTION", 0.05))
CHANNELS = int(os.environ.get("CHANNELS", 8))

if SMOKE:
    SEEDS       = range(2)
    MAX_EPOCHS  = 5
    PATIENCE    = 3
    CHANNELS    = 2             # smallest model = fastest smoke run, regardless of env
else:
    SEEDS       = range(15)     # 15 seeds: SEM ~0.0034 at sigma~0.013; enough to
    MAX_EPOCHS  = 100           # resolve the ~1-2 pp GE-vs-CNN gap (see chat rationale)
    PATIENCE    = 10

MIN_DELTA             = 1e-4    # minimum monitor-loss drop to count as improvement
TEST_FRACTION_MONITOR = 0.1     # cheap per-epoch early-stopping signal
TEST_FRACTION_FINAL   = 0.2     # 20% stratified test set for final + per-class accuracy

L             = 2
N_RINGS       = 6               # fixed for every GE-CNN (max for a 5x5 kernel)
GE_MAX_CHANNELS = 16            # skip GE-CNN above this width (ch=32 GE is far too heavy);
                                # CNN still runs at every channel width
GE_MAX_CHANNELS = int(os.environ.get("GE_MAX_CHANNELS", GE_MAX_CHANNELS))
KERNEL_SIZE   = 5
BATCH_SIZE    = 128
LR            = 1e-3
N_CLASSES     = 10
N_CONV_LAYERS = 2
CONV_PR_POOL  = 1
NUM_WORKERS   = 4
SAVE_SEED_WEIGHTS = True        # save the trained weights for every seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_ROOT = FAG_PROJEKT_DIR / "reports" / "param_datasize_2"
# Model is the TOP-LEVEL folder (CNN / GE_CNN) so each model's results are cleanly
# separated; the datasize x channels cell is a subpath under each model.
CELL_SUBPATH = Path(f"datasize_{TRAIN_FRACTION}") / f"channels_{CHANNELS}"

MODEL_NAMES = ["CNN", "GE_CNN"]

# Image size after padding (28 + 2*6 = 40) or raw (28). Computed without loading
# the dataset so re-imported DataLoader workers stay cheap.
from data import PADDING as PAD_PX
IMG_SIZE = 28 + 2 * PAD_PX if PADDING else 28

ARCH_KWARGS = dict(kernel_size=KERNEL_SIZE, img_size=IMG_SIZE, n_classes=N_CLASSES,
                   n_conv_layers=N_CONV_LAYERS, conv_pr_pool=CONV_PR_POOL, bias=True)


def build_model(name):
    if name == "CNN":
        m = CNN(channels=CHANNELS, in_features=1, **ARCH_KWARGS)
    elif name == "GE_CNN":
        m = GE_CNN(channels=CHANNELS, in_features=1, l=L, n_rings=N_RINGS, **ARCH_KWARGS)
    else:
        raise ValueError(name)
    m.name = name
    m(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE))   # materialise LazyLinear before counting/saving
    return m


def model_args(name):
    base = dict(kernel_size=KERNEL_SIZE, in_features=1, img_size=IMG_SIZE,
                n_conv_layers=N_CONV_LAYERS, conv_pr_pool=CONV_PR_POOL,
                channels=CHANNELS, n_classes=N_CLASSES, bias=True)
    if name == "GE_CNN":
        base.update(l=L, n_rings=N_RINGS)
    return base


def make_loaders(fraction, seed):
    """One train subset per seed (same subset & shuffle order shared by both models
    that seed = paired comparison). A non-shuffled eval loader for per-class on train."""
    train_ds = RotatedMNIST(split="train", rotated=True, fraction=fraction, padding=PADDING, seed=seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, generator=g)
    train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS)
    return train_loader, train_eval_loader, len(train_ds)


def train_early_stop(model, train_loader, monitor_loader, seed):
    """Train with early stopping on monitor loss; restore best-epoch weights.
    Returns the full per-epoch history and the best epoch index (1-based)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(DEVICE)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best = {"loss": float("inf"), "epoch": -1, "state": None}
    epochs_no_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = torch.zeros((), device=device, dtype=torch.float64)
        n_correct = torch.zeros((), device=device, dtype=torch.long)
        n_seen = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            bs = images.size(0)
            total_loss += loss.detach().double() * bs
            n_correct += (logits.argmax(dim=1) == labels).sum()
            n_seen += bs

        history["train_loss"].append((total_loss / n_seen).item())
        history["train_acc"].append((n_correct.double() / n_seen).item())

        val_loss, val_acc = evaluate(model, monitor_loader, show_progress=False, device=device)
        history["test_loss"].append(val_loss)
        history["test_acc"].append(val_acc)

        if val_loss < best["loss"] - MIN_DELTA:
            best = {"loss": val_loss, "epoch": epoch + 1,
                    "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return history, best["epoch"]


def per_class_accuracy(model, loader, device):
    """Per-class and overall accuracy on a loader."""
    model.eval()
    device = torch.device(device)
    correct = torch.zeros(N_CLASSES)
    total = torch.zeros(N_CLASSES)
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            for c in range(N_CLASSES):
                mask = labels == c
                total[c] += int(mask.sum())
                correct[c] += int((preds[mask] == c).sum())
    per_class = (correct / total.clamp(min=1)).tolist()
    overall = float(correct.sum() / total.sum()) if total.sum() > 0 else 0.0
    return {"per_class": per_class, "counts": total.int().tolist(), "overall": overall}


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def main():
    # Fixed test sets (same eval target for every model/seed).
    test_monitor_ds = RotatedMNIST(split="test", rotated=True, fraction=TEST_FRACTION_MONITOR, padding=PADDING, stratified=True)
    test_final_ds   = RotatedMNIST(split="test", rotated=True, fraction=TEST_FRACTION_FINAL, padding=PADDING, stratified=True)
    test_monitor_loader = DataLoader(test_monitor_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_final_loader   = DataLoader(test_final_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    assert test_monitor_ds.images.shape[-1] == IMG_SIZE, \
        f"IMG_SIZE mismatch: data is {test_monitor_ds.images.shape[-1]}, expected {IMG_SIZE}"

    print(f"Device: {DEVICE} | SMOKE={SMOKE} | datasize={TRAIN_FRACTION} | channels={CHANNELS} "
          f"| seeds={list(SEEDS)} | max_epochs={MAX_EPOCHS} | patience={PATIENCE} | img={IMG_SIZE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    cell = {"train_fraction": TRAIN_FRACTION, "channels": CHANNELS, "padding": PADDING,
            "l": L, "n_rings": N_RINGS, "kernel_size": KERNEL_SIZE, "lr": LR,
            "batch_size": BATCH_SIZE, "n_conv_layers": N_CONV_LAYERS, "conv_pr_pool": CONV_PR_POOL,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "min_delta": MIN_DELTA,
            "test_fraction_monitor": TEST_FRACTION_MONITOR, "test_fraction_final": TEST_FRACTION_FINAL,
            "seeds": list(SEEDS), "img_size": IMG_SIZE, "device": DEVICE, "smoke": SMOKE}

    for name in MODEL_NAMES:
        if name == "GE_CNN" and CHANNELS > GE_MAX_CHANNELS:
            print(f"  GE_CNN skipped at channels={CHANNELS} (too heavy; GE capped at ch={GE_MAX_CHANNELS})")
            continue
        model_dir = RESULTS_ROOT / name / CELL_SUBPATH
        model_dir.mkdir(parents=True, exist_ok=True)
        if (model_dir / "metrics.json").exists():
            print(f"  {name}: metrics.json already present -> skip (kept, not recomputed)")
            continue
        args = model_args(name)
        n_params = None
        per_seed = []

        for seed in SEEDS:
            train_loader, train_eval_loader, n_train = make_loaders(TRAIN_FRACTION, seed)
            model = build_model(name)
            if n_params is None:
                n_params = sum(p.numel() for p in model.parameters())
                print(f"\n{name}: {n_params:,} params (channels={CHANNELS})")

            history, best_epoch = train_early_stop(model, train_loader, test_monitor_loader, seed)

            final_loss, final_acc = evaluate(model, test_final_loader, show_progress=False, device=DEVICE)
            pc_train = per_class_accuracy(model, train_eval_loader, DEVICE)
            pc_test  = per_class_accuracy(model, test_final_loader, DEVICE)
            print(f"  [seed {seed}] best_epoch={best_epoch:3d}  test_acc={final_acc:.4f}  (n_train={n_train})")

            if SAVE_SEED_WEIGHTS:
                torch.save({"model_name": name, "model_args": args, "n_params": n_params,
                            "seed": seed, "best_epoch": best_epoch,
                            "state_dict": model.state_dict()},
                           model_dir / f"weights_seed{seed}.pth")

            per_seed.append({
                "seed": seed, "n_train": n_train, "best_epoch": best_epoch,
                "history": history,
                "final_test_loss": final_loss, "final_test_acc": final_acc,
                "per_class_train": pc_train, "per_class_test": pc_test,
            })

        test_accs = [r["final_test_acc"] for r in per_seed]
        acc_mean, acc_std = mean_std(test_accs)
        metrics = {
            "cell": cell, "model": name, "model_args": args, "n_params": n_params,
            "per_seed": per_seed,
            "aggregate": {"test_acc_mean": acc_mean, "test_acc_std": acc_std,
                          "test_acc_values": test_accs},
        }
        with open(model_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  {name}: test_acc {acc_mean:.4f} +/- {acc_std:.4f}  ->  {model_dir}")

    print(f"\nDone. Cell {CELL_SUBPATH} outputs under {RESULTS_ROOT}/<MODEL>/")


if __name__ == "__main__":
    main()
