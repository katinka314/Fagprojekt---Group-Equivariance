"""Quantify rotation invariance/equivariance of CNN vs GE-CNN.

Per image it produces: feature plots, class-score grids (logit + softmax), a softmax
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
    "CNN":    ROOT / "reports/unrot_test/10epoch_frac_0.03_15seeds/model_weights/CNN_smallrot_ch16_l2_r6_3pct_10epochs_seed10.pth",
    "GE_CNN": ROOT / "reports/unrot_test/10epoch_frac_0.03_15seeds/model_weights/GE_CNN_smallrot_ch8_l2_r6_3pct_10epochs_seed6.pth",
}
N_ROTATIONS   = 16     # number of angles, full circle (multiple of 4 -> 90-deg pairs exist)
TARGET_LAYER  = 6      # depth of the partial model used for the scalar metric table
NUM_IMAGES    = 4      # images to produce per-image plots/grids for
NUM_QUANT_IMG = 200    # images to average the scalar metrics over
N_GRID_IMG    = 200     # images to average the feature equivariance grid over
CHANNEL       = 1      # default feature channel (per-plot channel is set in main)
PADDING       = True
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR       = ROOT / "reports" / "quant_equi"
# -------------------------------------------------

ANGLES = [k * 360 / N_ROTATIONS for k in range(N_ROTATIONS)]
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Angle index sets: ORTHO = multiples of 90 deg (exact on a pixel grid), INTERP = the rest.
def is_ortho(a:float):
    return abs(a / 90 - round(a / 90)) < 1e-6
ORTHO        = [k for k, a in enumerate(ANGLES) if is_ortho(a)]
INTERP       = [k for k in range(len(ANGLES)) if k not in ORTHO]
FEAT_NONZERO = [k for k in range(len(ANGLES)) if k != 0]                  # drop trivial theta=0
FEAT_ORTHO   = [k for k in ORTHO if k != 0]                               # 90, 180, 270

# Pairwise grid masks: a pair (i,j) is an exact rotation when the angle difference is a
# multiple of 90 deg, i.e. (i-j) is a multiple of N_ROTATIONS/4 (covers e.g. 45->135 too).
_STEP90 = N_ROTATIONS // 4
_GI, _GJ = np.meshgrid(np.arange(N_ROTATIONS), np.arange(N_ROTATIONS), indexing="ij")
PAIR_EXACT  = ((_GI - _GJ) % _STEP90 == 0) & (_GI != _GJ)
PAIR_INTERP = ((_GI - _GJ) % _STEP90 != 0)
PAIR_ALL    = (_GI != _GJ)


def load_model(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model = {"CNN": CNN, "GE_CNN": GE_CNN}[ck["model_name"]](**ck["model_args"])
    model.load_state_dict(ck["state_dict"])
    model.eval().to(DEVICE)
    model.name = ck["model_name"]
    return model


def make_partial(model, target_layer=TARGET_LAYER): #Tager kun de led af modellen, som vi vil se igennem lag, for feature plot
    layers = list((model.features if model.name == "CNN" else model.model).children())
    return nn.Sequential(*layers[:min(target_layer, len(layers))])


def rotate_to_angles(img):                       # img [1,H,W] -> [N_angles,1,H,W]
    return torch.stack([rotate(img, a, interpolation=InterpolationMode.BILINEAR, fill=0)for a in ANGLES])


#metrics

def rms(O, center):                              # O [Rotation,Logit value] -> dispersion of rows across rotations
    X = (O - O.mean(dim=1, keepdim=True)) if center else O   # center: drop softmax constant-offset gauge
    V = ((X - X.mean(dim=0, keepdim=True)) ** 2).sum(dim=1).mean()   # = sum_c Var_angles(col_c)
    return float(V.sqrt())                       # RMS


def circular_mask(h, w, device):# laver en ciruklær maske, som kun regner fejlen inden for upåvirket cirkulære radius
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    r = torch.sqrt((ys - (h - 1) / 2) ** 2 + (xs - (w - 1) / 2) ** 2)
    return (r <= min((h - 1) / 2, (w - 1) / 2)).float()


def masked_rel_l2(a, b, mask):# a,b [C,h,w] -> relative L2 inside mask
    m = mask[None] #[1, 32, 32]
    num = (((a - b) * m) ** 2).sum().sqrt() #Fjerner hjørner med mask l_2 norm
    den = ((b * m) ** 2).sum().sqrt() + 1e-12 # l2 dist  af a og b
    return float(num / den) #normalized/ relativ output


def feature_equivariance(partial: nn.Sequential, img):  # img [1,H,W] -> per-angle rel-L2( f(rho x), rho f(x) )
    with torch.no_grad():
        f0 = partial(img[None]).abs()[0]         # f(x): reference feature of the unrotated image
    mask = circular_mask(*f0.shape[-2:], f0.device)
    errs = []
    for a in ANGLES:
        x_rot = rotate(img, a, interpolation=InterpolationMode.BILINEAR, fill=0)
        with torch.no_grad():
            f_rot = partial(x_rot[None]).abs()[0]  # f(rho x)
        ref = rotate(f0, a, interpolation=InterpolationMode.BILINEAR, fill=0)   # rho f(x)
        errs.append(masked_rel_l2(f_rot, ref, mask))
    return np.array(errs)


def grid_means(D):                               # D [R,R] -> [exact, interp, all] means over angle-difference classes
    D = np.asarray(D, dtype=float)
    return [float(D[PAIR_EXACT].mean()), float(D[PAIR_INTERP].mean()), float(D[PAIR_ALL].mean())]


def feature_grid(partial, img):                  # img [1,H,W] -> [R,R] rel-L2( rho_{j-i} f(rho_i x), f(rho_j x) )
    with torch.no_grad():
        feats = [partial(r[None]).abs()[0] for r in rotate_to_angles(img).to(DEVICE)]
    mask = circular_mask(*feats[0].shape[-2:], feats[0].device)
    n = len(ANGLES)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            aligned = rotate(feats[i], ANGLES[j] - ANGLES[i], interpolation=InterpolationMode.BILINEAR, fill=0)
            D[i, j] = masked_rel_l2(aligned, feats[j], mask)
    return D


#plots

def feats_at(model, target_layer, img):          # img [1,H,W] -> list of [C,h,w] (abs |f|), one per global angle
    partial = make_partial(model, target_layer)
    rots = rotate_to_angles(img).to(DEVICE)
    with torch.no_grad():
        return [partial(r[None]).abs()[0] for r in rots]


def plot_feature_grid(model, name, target_layer, channel, img, img_idx, angles=(0, 22.5, 45,  90)):
    feats = feats_at(model, target_layer, img)
    idxs  = [min(range(len(ANGLES)), key=lambda k: abs(ANGLES[k] - a)) for a in angles]
    ch0   = feats[0][channel].cpu()
    bg    = float(ch0.min())                      # fill rotated-in corners with the background value
    ref   = torch.stack([rotate(ch0[None], a, interpolation=InterpolationMode.BILINEAR, fill=bg)
                         for a in ANGLES])
    n = len(angles)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    fig.suptitle(rf"{name}  |  layer {target_layer}  |  $|f|$ channel {channel}", fontsize=14)
    for col, (a, gi) in enumerate(zip(angles, idxs)):
        ax[0, col].imshow(ref[gi, 0].numpy(), cmap="viridis"); ax[0, col].axis("off")
        ax[0, col].set_title(rf"$\rho(g)\,f(x)$  {a:.0f}°")
        ax[1, col].imshow(feats[gi][channel].cpu().numpy(), cmap="viridis"); ax[1, col].axis("off")
        ax[1, col].set_title(rf"$f(\rho(g)\,x)$  {a:.0f}°")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / f"{name}_img{img_idx}_L{target_layer}_ch{channel}_feature_plot.png", dpi=200)
    plt.close(fig)


def plot_class_scores(mat, name, idx, kind, cls):     # mat [N_angles,10]; cls = true class of the image
    plt.figure(figsize=(10, 4))
    plt.imshow(mat.cpu().numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label=kind); plt.xlabel("class"); plt.ylabel("rotation")
    plt.xticks(range(10), CLASS_NAMES, rotation=90)
    plt.yticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES])
    plt.title(f"{name}: {kind} scores for image class: {cls}"); plt.tight_layout()
    plt.savefig(OUT_DIR / f"{name}_img{idx}_classscores_{kind}.png", dpi=200); plt.close()


def plot_invariance_grid(D, name, vmax=None):    # mean softmax invariance grid ||p_i - p_j||
    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="viridis", vmin=0, vmax=vmax); plt.colorbar(label=r"Dissimilarity")
    plt.xticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES], rotation=90)
    plt.yticks(range(len(ANGLES)), [f"{a:.0f}" for a in ANGLES])
    plt.xlabel("angle j"); plt.ylabel("angle i")
    plt.title(f"{name}: softmax invariance grid)")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"{name}_invariance_grid.png", dpi=200); plt.close()


def plot_feature_diffgrid(model, name, target_layer, imgs):  # MEAN equivariance grid over imgs
    n = len(ANGLES)
    D = np.zeros((n, n))
    for img in imgs:                                  # accumulate the per-image grid, then average
        feats = feats_at(model, target_layer, img)
        mask  = circular_mask(*feats[0].shape[-2:], feats[0].device)
        for i in range(n):
            for j in range(n):
                aligned = rotate(feats[i], ANGLES[j] - ANGLES[i], interpolation=InterpolationMode.BILINEAR, fill=0)
                D[i, j] += masked_rel_l2(aligned, feats[j], mask)
    D /= len(imgs)
    plt.figure(figsize=(6, 5))
    plt.imshow(D, cmap="viridis", vmin=0, vmax=1); plt.colorbar(label=r"Equivariance dissimilarity")
    plt.xticks(range(n), [f"{a:.0f}" for a in ANGLES], rotation=90)
    plt.yticks(range(n), [f"{a:.0f}" for a in ANGLES])
    plt.xlabel("angle j"); plt.ylabel("angle i")
    plt.title(rf"{name}: feature equivariance grid (layer {target_layer})")
    plt.tight_layout(); plt.savefig(OUT_DIR / f"{name}_feature_diffgrid_L{target_layer}.png", dpi=200)


Z95 = 1.959964   # normal-approx 95% half-width (n=200 -> t_{.975,199}=1.972, <1% larger)


def ms(values):
    a = np.asarray(values, dtype=float)
    n = a.size
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    ci95 = float(Z95 * std / np.sqrt(n)) if n > 1 else 0.0   # 95% CI half-width on the mean
    return [float(a.mean()), std, ci95]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data   = RotatedMNIST(split="test", rotated=False, padding=PADDING)   # unrotated; we rotate ourselves
    models = {name: load_model(path) for name, path in WEIGHTS.items()}

    # --- Per-image plots: feature plots + class-score grids for NUM_IMAGES images ---
    # The same (model, layer, channel) feature combos are shown for every image.
    feature_plots = [("GE_CNN", 2, 3), ("GE_CNN", 6, 7), ("CNN", 2, 3), ("CNN", 6, 15)]
    for x in range(NUM_IMAGES):
        img = data[x][0]
        cls = CLASS_NAMES[int(data[x][1])]                                # true class of this image
        for name, layer, ch in feature_plots:
            plot_feature_grid(models[name], name, layer, ch, img, x)
        rots = rotate_to_angles(img).to(DEVICE)
        for name, model in models.items():
            with torch.no_grad():
                logits = torch.stack([model(r[None])[0] for r in rots])
            plot_class_scores(torch.softmax(logits, dim=1), name, x, "softmax", cls)

    # --- Feature equivariance grids at layer 2 and 6, both models (mean over images) ---
    grid_imgs = [data[i][0] for i in range(N_GRID_IMG)]
    for name, model in models.items():
        for layer in (2, 6):
            plot_feature_diffgrid(model, name, layer, grid_imgs)

    # --- Scalar metrics + mean softmax invariance grids (table uses global TARGET_LAYER) ---
    summary = {}
    inv_grids = {}
    for name, model in models.items():
        partial  = make_partial(model, TARGET_LAYER)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name} | params={n_params:,} | layers={TARGET_LAYER} | angles={N_ROTATIONS} ===")

        acc = {m: {s: [] for s in ("exact", "interp", "all")}
               for m in ("logit", "softmax", "feature")}
        inv_grid_sum = np.zeros((N_ROTATIONS, N_ROTATIONS))
        for i in range(NUM_QUANT_IMG):                    # scalar metrics over many images
            img = data[i][0].to(DEVICE)
            rots = rotate_to_angles(img).to(DEVICE)
            with torch.no_grad():
                logits = torch.stack([model(r[None])[0] for r in rots])
            soft = torch.softmax(logits, dim=1)
            logits_c = logits - logits.mean(dim=1, keepdim=True)   # drop the logit constant-offset gauge

            grids = {"logit":   torch.cdist(logits_c, logits_c).cpu().numpy(),
                     "softmax": torch.cdist(soft, soft).cpu().numpy(),
                     "feature": feature_grid(partial, img)}
            inv_grid_sum += grids["softmax"]
            for m, D in grids.items():
                e, ip, a = grid_means(D)
                acc[m]["exact"].append(e)
                acc[m]["interp"].append(ip)
                acc[m]["all"].append(a)
        inv_grids[name] = inv_grid_sum / NUM_QUANT_IMG

        summary[name] = {"n_params": n_params,
                         **{m: {s: ms(acc[m][s]) for s in acc[m]} for m in acc}}
        for m in ("logit", "softmax", "feature"):
            e, ip, a = (summary[name][m][s] for s in ("exact", "interp", "all"))
            print(f"  {m:8s}  all={a[0]:.4f}+/-{a[2]:.4f}  exact90={e[0]:.4f}+/-{e[2]:.4f}  interp={ip[0]:.4f}+/-{ip[2]:.4f}  (95% CI)")

    inv_vmax = max(float(D.max()) for D in inv_grids.values())
    for name, D in inv_grids.items():
        plot_invariance_grid(D, name, vmax=inv_vmax)

    json.dump(summary, open(OUT_DIR / "invariance_summary.json", "w"), indent=2)
    print(f"\nDone. Plots + summary in {OUT_DIR}")



if __name__ == "__main__":
    main()
