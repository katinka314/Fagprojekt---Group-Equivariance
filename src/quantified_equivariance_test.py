"""Quantify rotation invariance/equivariance of CNN vs GE-CNN.

Per image it produces: feature plots, class-score grids (logit + softmax), a logit
invariance diff-grid, and a feature equivariance diff-grid. Over the test set it
reports three scalar metrics, each split into 90-deg-multiple angles (exact rotations,
the architectural floor) vs interpolated angles (the bilinear-interpolation floor):
  - logit_rms     : variance of logits across rotations  (invariance)
  - softmax_rms   : same on softmax probabilities (bounded, comparable across models)
  - feature_equiv : rel-L2 between f(rho x) and rho f(x) (equivariance of features)

Model parameters are read back from each checkpoint's `model_args`.
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
N_ROTATIONS   = 16     # number of angles, full circle (multiple of 4 -> 90-deg pairs exist)
TARGET_LAYER  = 2      # depth of the partial model for the feature plots / equivariance
NUM_IMAGES    = 1      # images to produce per-image plots/grids for
NUM_QUANT_IMG = 100    # images to average the scalar metrics over
CHANNEL       = -1     # feature channel to visualise
PADDING       = True
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR       = ROOT / "reports" / "quant_equi" / "equivariance_test"
# -------------------------------------------------

ANGLES = [k * 360 / N_ROTATIONS for k in range(N_ROTATIONS)]
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Angle index sets: ORTHO = multiples of 90 deg (exact on a pixel grid), INTERP = the rest.
def _is_ortho(a):
    return abs(a / 90 - round(a / 90)) < 1e-6
ORTHO        = [k for k, a in enumerate(ANGLES) if _is_ortho(a)]          # 0, 90, 180, 270
INTERP       = [k for k in range(len(ANGLES)) if k not in ORTHO]
FEAT_NONZERO = [k for k in range(len(ANGLES)) if k != 0]                  # drop trivial theta=0
FEAT_ORTHO   = [k for k in ORTHO if k != 0]                               # 90, 180, 270


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


# ---- metrics ----

def rms(O, center):                              # O [N,C] -> dispersion of rows across rotations
    X = (O - O.mean(dim=1, keepdim=True)) if center else O   # center: drop softmax constant-offset gauge
    V = ((X - X.mean(dim=0, keepdim=True)) ** 2).sum(dim=1).mean()   # = sum_c Var_angles(col_c)
    return float(V.sqrt())                       # RMS


def circular_mask(h, w, device):                 # inscribed disk: stays in-frame under rotation
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    r = torch.sqrt((ys - (h - 1) / 2) ** 2 + (xs - (w - 1) / 2) ** 2)
    return (r <= min((h - 1) / 2, (w - 1) / 2)).float()


def masked_rel_l2(a, b, mask):                   # a,b [C,h,w] -> relative L2 inside mask
    m = mask[None]
    num = (((a - b) * m) ** 2).sum().sqrt()
    den = ((b * m) ** 2).sum().sqrt() + 1e-12
    return float(num / den)


def feature_equivariance(partial, img):          # img [1,H,W] -> per-angle rel-L2( f(rho x), rho f(x) )
    with torch.no_grad():
        f0 = partial(img[None]).abs()[0]         # f(x): reference feature of the unrotated image
    mask = circular_mask(*f0.shape[-2:], f0.device)
    errs = []
    for a in ANGLES:
        x_rot = rotate(img, a, interpolation=InterpolationMode.BILINEAR, fill=0)
        with torch.no_grad():
            f_rot = partial(x_rot[None]).abs()[0]                # f(rho x)
        ref = rotate(f0, a, interpolation=InterpolationMode.BILINEAR, fill=0)   # rho f(x)
        errs.append(masked_rel_l2(f_rot, ref, mask))
    return np.array(errs)


# ---- plots ----

def plot_feature_grid(feats, name, idx):         # feats: list of [C,h,w] (abs), one per angle
    ref = rotate_to_angles(feats[0][CHANNEL][None].cpu())        # rho f(x): rotate the unrotated feature
    n = len(feats)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    for i, f in enumerate(feats):
        ax[0, i].imshow(ref[i, 0].numpy(), cmap="gray"); ax[0, i].axis("off")
        ax[0, i].set_title(f"rho(g)f(x)  {ANGLES[i]:.0f}")
        ax[1, i].imshow(f[CHANNEL].cpu().numpy(), cmap="gray"); ax[1, i].axis("off")
        ax[1, i].set_title(f"f(rho(g)x)  {ANGLES[i]:.0f}")
    fig.tight_layout(); fig.savefig(OUT_DIR / f"{name}_feature_plot_img{idx}.png", dpi=200); plt.close(fig)


def plot_class_scores(mat, name, idx, kind):     # mat [N_angles,10]
    plt.figure(figsize=(10, 4))
    plt.imshow(mat.cpu().numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label=kind); plt.xlabel("class"); plt.ylabel("rotation")
    plt.xticks(range(10), CLASS_NAMES, rotation=90)
    plt.yticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES])
    plt.title(f"{name}: {kind} scores"); plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_classscores_{kind}_img{idx}.png", dpi=200); plt.close()


def plot_logit_diffgrid(mat, name, idx):         # invariance: ||o_i - o_j|| (no alignment)
    D = torch.cdist(mat, mat).cpu().numpy()
    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="viridis"); plt.colorbar(label="||logits_i - logits_j||")
    plt.xticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES], rotation=90)
    plt.yticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES])
    plt.xlabel("angle j"); plt.ylabel("angle i"); plt.title(f"{name}: logit invariance grid")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"{name}_diffgrid_img{idx}.png", dpi=200); plt.close()


def plot_feature_diffgrid(feats, name, idx):     # equivariance: rel-L2( rho_{j-i} f(rho_i x), f(rho_j x) )
    n = len(feats)
    mask = circular_mask(*feats[0].shape[-2:], feats[0].device)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            aligned = rotate(feats[i], ANGLES[j] - ANGLES[i], interpolation=InterpolationMode.BILINEAR, fill=0)
            D[i, j] = masked_rel_l2(aligned, feats[j], mask)
    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="viridis"); plt.colorbar(label="rel-L2( rho(f_i), f_j )")
    plt.xticks(range(n), [f"{a:.0f}" for a in ANGLES], rotation=90)
    plt.yticks(range(n), [f"{a:.0f}" for a in ANGLES])
    plt.xlabel("angle j"); plt.ylabel("angle i"); plt.title(f"{name}: feature equivariance grid")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"{name}_feature_diffgrid_img{idx}.png", dpi=200); plt.close()


def ms(values):
    a = np.asarray(values, dtype=float)
    return [float(a.mean()), float(a.std())]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = RotatedMNIST(split="test", rotated=False, padding=PADDING)   # unrotated; we rotate ourselves
    summary = {}

    for name, path in WEIGHTS.items():
        model = load_model(path)
        partial = make_partial(model)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name} | params={n_params:,} | layers={TARGET_LAYER} | angles={N_ROTATIONS} ===")

        for i in range(NUM_IMAGES):                       # per-image plots
            rots = rotate_to_angles(data[i][0]).to(DEVICE)
            with torch.no_grad():
                feats  = [partial(r[None]).abs()[0] for r in rots]
                logits = torch.stack([model(r[None])[0] for r in rots])
            soft = torch.softmax(logits, dim=1)
            plot_feature_grid(feats, name, i)
            plot_class_scores(logits, name, i, "logit")
            plot_class_scores(soft, name, i, "softmax")
            plot_logit_diffgrid(logits, name, i)          # invariance
            plot_feature_diffgrid(feats, name, i)         # equivariance

        acc = {m: {s: [] for s in ("all", "ortho90", "interp")}
               for m in ("logit", "softmax", "feature")}
        for i in range(NUM_QUANT_IMG):                    # scalar metrics over many images
            img = data[i][0].to(DEVICE)
            rots = rotate_to_angles(img).to(DEVICE)
            with torch.no_grad():
                logits = torch.stack([model(r[None])[0] for r in rots])
            soft = torch.softmax(logits, dim=1)
            fe = feature_equivariance(partial, img)

            acc["logit"]["all"].append(rms(logits, center=True))
            acc["logit"]["ortho90"].append(rms(logits[ORTHO], center=True))
            acc["logit"]["interp"].append(rms(logits[INTERP], center=True))
            acc["softmax"]["all"].append(rms(soft, center=False))
            acc["softmax"]["ortho90"].append(rms(soft[ORTHO], center=False))
            acc["softmax"]["interp"].append(rms(soft[INTERP], center=False))
            acc["feature"]["all"].append(fe[FEAT_NONZERO].mean())
            acc["feature"]["ortho90"].append(fe[FEAT_ORTHO].mean())
            acc["feature"]["interp"].append(fe[INTERP].mean())

        summary[name] = {"n_params": n_params,
                         **{m: {s: ms(acc[m][s]) for s in acc[m]} for m in acc}}
        for m in ("logit", "softmax", "feature"):
            a, o, ip = (summary[name][m][s] for s in ("all", "ortho90", "interp"))
            print(f"  {m:8s}  all={a[0]:.4f}  ortho90={o[0]:.4f}  interp={ip[0]:.4f}")

    json.dump(summary, open(OUT_DIR / "invariance_summary.json", "w"), indent=2)
    print(f"\nDone. Plots + summary in {OUT_DIR}")


if __name__ == "__main__":
    main()
