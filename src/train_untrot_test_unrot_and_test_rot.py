"""Evaluate the saved (final-epoch) CNN and GE-CNN models on three test sets over
all seeds: unrotated, rotated by an exact multiple of 90 degrees (no interpolation),
and randomly rotated. No training: weights are loaded from
reports/unrot_test/10epoch_frac_0.03_15seeds/model_weights/.

Reports mean +/- std final test accuracy per model on each test set, prints the
numbers, and saves a bar plot plus a JSON summary.
"""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless: compute nodes have no display
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from data import RotatedMNIST, rotate_90
from train import evaluate
from models.Model import CNN, GE_CNN


# Config

RUN_DIR       = FAG_PROJEKT_DIR / "reports" / "unrot_test" / "10epoch_frac_0.03_15seeds"
WEIGHTS_DIR   = RUN_DIR / "model_weights"
OUT_DIR       = RUN_DIR / "final_eval_unrot_vs_rot"
PADDING       = True
TEST_FRACTION = 0.4  # 1.0 = full test set
BATCH_SIZE    = 128
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CLASSES = {"CNN": CNN, "GE_CNN": GE_CNN}
MODEL_NAMES   = ["CNN", "GE_CNN"]


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = MODEL_CLASSES[ck["model_name"]](**ck["model_args"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(DEVICE)
    model.name = ck["model_name"]
    return model


def seed_of(path):
    m = re.search(r"seed(\d+)", path.stem)
    return int(m.group(1)) if m else -1




SPLITS = ["unrot", "rot90", "rot"]
SPLIT_COLOR = {"unrot": "tab:green", "rot90": "tab:blue", "rot": "tab:red"}
SPLIT_LABEL = {"unrot": "test (unrot)", "rot90": "test (rot 90)", "rot": "test (rot)"}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    test_unr = RotatedMNIST(split="test", rotated=False, fraction=TEST_FRACTION, padding=PADDING)
    test_rot = RotatedMNIST(split="test", rotated=True,  fraction=TEST_FRACTION, padding=PADDING)
    test_90  = RotatedMNIST(split="test", rotated=False, fraction=TEST_FRACTION, padding=PADDING)
    test_90.images, test_90.angles = rotate_90(test_90.images)

    loaders = {
        "unrot": DataLoader(test_unr, batch_size=BATCH_SIZE, shuffle=False),
        "rot90": DataLoader(test_90,  batch_size=BATCH_SIZE, shuffle=False),
        "rot":   DataLoader(test_rot, batch_size=BATCH_SIZE, shuffle=False),
    }
    print(f"Device: {DEVICE} | test images: unrot={len(test_unr)} rot90={len(test_90)} "
          f"rot={len(test_rot)} | frac={TEST_FRACTION}")

    acc = {n: {s: [] for s in SPLITS} for n in MODEL_NAMES}
    for name in MODEL_NAMES:
        paths = sorted(WEIGHTS_DIR.glob(f"{name}_*seed*.pth"), key=seed_of)
        print(f"\n{name}: {len(paths)} checkpoints")
        for p in paths:
            model = load_model(p)
            accs = {s: evaluate(model, loaders[s], show_progress=False, device=DEVICE)[1] for s in SPLITS}
            for s in SPLITS:
                acc[name][s].append(accs[s])
            print(f"  seed {seed_of(p):2d}: " + "  ".join(f"{s}={accs[s]:.4f}" for s in SPLITS))

    # Aggregate
    summary = {}
    print("\n=== Final test accuracy (mean +/- std over seeds) ===")
    for name in MODEL_NAMES:
        summary[name] = {}
        for split in SPLITS:
            vals = np.asarray(acc[name][split], dtype=float)
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
            summary[name][split] = {"mean": mean, "std": std, "n_seeds": int(vals.size),
                                    "values": vals.tolist()}
            print(f"  {name:7s} {split:6s}: {mean:.4f} +/- {std:.4f}  (n={vals.size})")

    with open(OUT_DIR / "final_eval_summary.json", "w") as f:
        json.dump({"test_fraction": TEST_FRACTION, "padding": PADDING, "device": DEVICE,
                   "summary": summary}, f, indent=2)

    # Bar plot: unrot / rot90 / rot, per model, mean +/- std
    x = np.arange(len(MODEL_NAMES)); w = 0.27
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, s in enumerate(SPLITS):
        m = [summary[n][s]["mean"] for n in MODEL_NAMES]
        e = [summary[n][s]["std"]  for n in MODEL_NAMES]
        ax.bar(x + (k - 1) * w, m, w, yerr=e, capsize=4, label=SPLIT_LABEL[s], color=SPLIT_COLOR[s])
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    n_seeds = summary[MODEL_NAMES[0]]["unrot"]["n_seeds"]
    ax.set_title(f"Final test accuracy (mean +/- std over {n_seeds} seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_accuracy_unrot_vs_rot.png", dpi=150)
    plt.close(fig)

    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
