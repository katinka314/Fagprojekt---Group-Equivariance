"""Recreate the "train unrotated, test rotated" accuracy-curve figure, but draw a
95% confidence interval of the mean instead of a +/- 1 std band.

Reads the precomputed per-epoch statistics from summary_mean_std.json (it stores
the per-seed mean and std at each epoch for train_acc / test_acc of CNN and
GE_CNN over 15 seeds). The CI half-width at each epoch is

    t_{(1+conf)/2, n-1} * std / sqrt(n)

with n = number of seeds and std the stored per-seed (sample) standard deviation.

No training and no model loading: this is a pure replotting step.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless / save-to-file only
import matplotlib.pyplot as plt
from scipy import stats


# Config

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = FAG_PROJEKT_DIR / "reports" / "unrot_test" / "10epoch_frac_0.03_15seeds"
SUMMARY_PATH = RUN_DIR / "summary_mean_std.json"
OUT_PATH = RUN_DIR / "plots" / "accuracy_curves_mean_ci_swapped_colors.png"

CONFIDENCE = 0.95
MODEL_NAMES = ["CNN", "GE_CNN"]
# Colors match the original figure (CNN orange, GE_CNN blue).
COLORS = {"CNN": "tab:orange", "GE_CNN": "tab:blue"}
BAND_ALPHA = 0.18


def ci_halfwidth(std: np.ndarray, n: int, confidence: float) -> np.ndarray:
    """95% (by default) CI half-width of the mean, per epoch, via Student-t."""
    t_crit = stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1)
    return t_crit * std / np.sqrt(n)


def main() -> None:
    import json

    with open(SUMMARY_PATH) as f:
        d = json.load(f)

    n_seeds = len(d["config"]["seeds"])
    n_epochs = d["config"]["n_epochs"]
    n_params = d["n_params"]
    epochs = np.arange(1, n_epochs + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in MODEL_NAMES:
        c = COLORS[name]
        lbl = f"{name} ({n_params[name]:,}p)"

        # train (unrot): solid line; test (rot): dashed line
        for acc_key, style, suffix in [
            ("train_acc", "-", "train (unrot)"),
            ("test_acc", "--", "test (rot)"),
        ]:
            mean = np.asarray(d[name][acc_key]["mean"], dtype=float)
            std = np.asarray(d[name][acc_key]["std"], dtype=float)
            half = ci_halfwidth(std, n_seeds, CONFIDENCE)
            ax.plot(epochs, mean, color=c, linestyle=style, label=f"{lbl} {suffix}")
            ax.fill_between(epochs, mean - half, mean + half, color=c, alpha=BAND_ALPHA)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    pct = int(round(CONFIDENCE * 100))
    ax.set_title(f"Accuracy: mean +/- {pct}% CI over {n_seeds} seeds")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
