"""Quantify rotation invariance of CNN vs GE-CNN: feature plots, class-score grids,
pairwise angle-difference grids, and a scalar logit-variance metric over the test set.

Model weights come from small_rotation_train.py (the model parameters are read back
from each checkpoint's `model_args`). Plots land in reports/plots/equivariance_test/.
"""
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rotate, InterpolationMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from data import RotatedMNIST
from models.Model import CNN, GE_CNN

# ---------------- FREE PARAMETERS ----------------
WEIGHTS = {
    "CNN":    ROOT / "models/model_weights/CNN_smallrot_ch4_l2_r6_3pct.pth",
    "GE_CNN": ROOT / "models/model_weights/GE_CNN_smallrot_ch4_l2_r6_3pct.pth",
}
N_ROTATIONS   = 16      # number of angles, full circle (multiple of 4 -> 90-deg pairs exist)
TARGET_LAYER  = 2      # depth of the partial model for the feature plots
NUM_IMAGES    = 1      # images to produce per-image plots/grids for
NUM_QUANT_IMG = 100    # images to average the variance metric over
CHANNEL       = -1      # feature channel to visualise
PADDING       = True
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR       = ROOT / "reports" / "quant_equi" / "equivariance_test"
# -------------------------------------------------

ANGLES = [k * 360 / N_ROTATIONS for k in range(N_ROTATIONS)]
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = {"CNN": CNN, "GE_CNN": GE_CNN}[ck["model_name"]](**ck["model_args"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(DEVICE)
    model.name = ck["model_name"]
    return model


def make_partial(model):
    layers = list((model.features if model.name == "CNN" else model.model).children())
    return nn.Sequential(*layers[:min(TARGET_LAYER, len(layers))])


def rotate_to_angles(img):                       # img [1,H,W] -> [N_angles,1,H,W]
    return torch.stack([rotate(img, a, interpolation=InterpolationMode.BILINEAR, fill=0)
                        for a in ANGLES])


def logit_rms(O):                                # O [N_angles,10] -> scalar invariance error
    Oc = O - O.mean(dim=1, keepdim=True)         # remove softmax constant-offset gauge
    V = ((Oc - Oc.mean(dim=0, keepdim=True)) ** 2).sum(dim=1).mean()   # = sum_c Var_angles(logit_c)
    return float(V.sqrt())                       # RMS, in logit units


def plot_feature_grid(feats, name, idx):         # feats: list of [C,h,w] (abs), one per angle
    ref = rotate_to_angles(feats[0][CHANNEL][None].cpu())  # rho * f(x): rotate the unrotated feature
    n = len(feats)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    for i, f in enumerate(feats):
        ax[0, i].imshow(ref[i, 0].numpy(), cmap="gray"); ax[0, i].axis("off")
        ax[0, i].set_title(f"rho(g)f(x)  {ANGLES[i]:.0f}")
        ax[1, i].imshow(f[CHANNEL].cpu().numpy(), cmap="gray"); ax[1, i].axis("off")
        ax[1, i].set_title(f"f(rho(g)x)  {ANGLES[i]:.0f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{name}_feature_plot_img{idx}.png", dpi=200)
    plt.close(fig)


def plot_class_scores(logits, name, idx):        # logits [N_angles,10]
    plt.figure(figsize=(10, 4))
    plt.imshow(logits.cpu().numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="logit"); plt.xlabel("class"); plt.ylabel("rotation")
    plt.xticks(range(10), CLASS_NAMES, rotation=90)
    plt.yticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES])
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_classscores_img{idx}.png", dpi=200)
    plt.close()


def plot_diff_grid(logits, name, idx):           # pairwise angle x angle ||o_i - o_j||
    D = torch.cdist(logits, logits).cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="viridis"); plt.colorbar(label="||logits_i - logits_j||")
    plt.xticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES], rotation=90)
    plt.yticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES])
    plt.xlabel("angle j"); plt.ylabel("angle i"); plt.title(f"{name}: rotation diff grid")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_diffgrid_img{idx}.png", dpi=200)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = RotatedMNIST(split="test", rotated=False, padding=PADDING)   # unrotated; we rotate ourselves
    summary = {}
    for name, path in WEIGHTS.items():
        model = load_model(path)
        partial = make_partial(model)
        print(f"\n=== {name} | layers={TARGET_LAYER} | angles={N_ROTATIONS} ===")

        for i in range(NUM_IMAGES):                       # per-image plots
            rots = rotate_to_angles(data[i][0]).to(DEVICE)
            with torch.no_grad():
                feats  = [partial(r[None]).abs()[0] for r in rots]
                logits = torch.stack([model(r[None])[0] for r in rots])
            plot_feature_grid(feats, name, i)
            plot_class_scores(logits, name, i)
            plot_diff_grid(logits, name, i)

        rms = []                                          # scalar metric over many images
        for i in range(NUM_QUANT_IMG):
            rots = rotate_to_angles(data[i][0]).to(DEVICE)
            with torch.no_grad():
                logits = torch.stack([model(r[None])[0] for r in rots])
            rms.append(logit_rms(logits))
        rms = np.array(rms)
        summary[name] = {"logit_rms_mean": float(rms.mean()), "logit_rms_std": float(rms.std())}
        print(f"{name}: logit-RMS invariance error = {rms.mean():.4f} +/- {rms.std():.4f}")

    json.dump(summary, open(OUT_DIR / "invariance_summary.json", "w"), indent=2)
    print(f"\nDone. Plots + summary in {OUT_DIR}")


if __name__ == "__main__":
    main()
