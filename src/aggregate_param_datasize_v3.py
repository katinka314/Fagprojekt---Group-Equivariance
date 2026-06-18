"""Collect the independent per-cell results of the datasize x channels grid into
one array, ready for plotting. Run AFTER all 30 jobs finish.

Walks reports/param_datasize_2/datasize_<f>/channels_<c>/<model>/metrics.json and
builds an accuracy array of shape (n_fractions, n_channels, n_models, n_seeds),
plus parameter counts per (channels, model). Saves a .pt and a .npz.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

ROOT = FAG_PROJEKT_DIR / "reports" / "param_datasize_2"

FRACTIONS = [0.005, 0.01, 0.03, 0.05, 0.1]
CHANNELS  = [2, 4, 8, 16, 32]
MODELS    = ["CNN", "GE_CNN"]   # top-level folder per model


def main():
    n_seeds = None
    acc = None          # (frac, channels, model, seed) of final test accuracy
    n_params = np.full((len(CHANNELS), len(MODELS)), np.nan)
    missing = []

    for fi, f in enumerate(FRACTIONS):
        for ci, c in enumerate(CHANNELS):
            for mi, m in enumerate(MODELS):
                path = ROOT / m / f"datasize_{f}" / f"channels_{c}" / "metrics.json"
                if not path.exists():
                    missing.append(str(path.relative_to(ROOT)))
                    continue
                metrics = json.loads(path.read_text())
                vals = metrics["aggregate"]["test_acc_values"]
                if n_seeds is None:
                    n_seeds = len(vals)
                    acc = np.full((len(FRACTIONS), len(CHANNELS), len(MODELS), n_seeds), np.nan)
                acc[fi, ci, mi, :len(vals)] = vals
                n_params[ci, mi] = metrics["n_params"]

    if acc is None:
        print("No metrics found. Did the jobs run?")
        return

    out = {
        "fractions": FRACTIONS, "channels": CHANNELS, "models": MODELS,
        "n_seeds": n_seeds,
        "accuracy": acc,                       # (frac, channels, model, seed)
        "accuracy_mean": np.nanmean(acc, axis=-1),
        "accuracy_std": np.nanstd(acc, axis=-1, ddof=1),
        "n_params": n_params,                  # (channels, model)
    }
    torch.save(out, ROOT / "grid_results.pt")
    np.savez(ROOT / "grid_results.npz",
             accuracy=acc, accuracy_mean=out["accuracy_mean"],
             accuracy_std=out["accuracy_std"], n_params=n_params,
             fractions=np.array(FRACTIONS), channels=np.array(CHANNELS),
             models=np.array(MODELS))

    print(f"Saved grid_results.pt / .npz  shape={acc.shape} (frac, channels, model, seed)")
    if missing:
        print(f"WARNING: {len(missing)} cells missing (jobs not finished or failed):")
        for p in missing:
            print(f"  {p}")


if __name__ == "__main__":
    main()
