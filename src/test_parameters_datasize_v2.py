"""Parameter- and data-efficiency sweeps: GE-CNN vs CNN, multi-seed, on rotated MNIST.

Experiment A (param vs param): for a range of CNN channel sizes, match a GE-CNN to
the same parameter budget and compare test accuracy. Shows how much capacity
equivariance saves at equal parameter count.

Experiment B (datasize): one parameter-matched pair, trained at increasing data
fractions. Shows how little data the equivariant model needs.

Both averaged over SEEDS so plots carry mean +/- std. Runs headless; designed to be
queued on the HPC and left overnight (incremental saving + per-config error handling).
"""

import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from data import RotatedMNIST
from train import train_loop, evaluate
from models.Model import GE_CNN, CNN


# Config

SMOKE   = True     # True = tiny local sanity run, False = full HPC run
PADDING = True

if SMOKE:
    SEEDS               = range(2)
    N_EPOCHS            = 2
    TRAIN_FRACTION_A    = 0.02
    TRAIN_FRACTIONS     = [0.01, 0.02, 0.05]
    CNN_CHANNELS_A      = [4, 8]
    CNN_CHANNELS_B      = 8
    TEST_FRACTION_CURVE = 0.05
    TEST_FRACTION_FINAL = 0.1
else:
    SEEDS               = range(5)
    N_EPOCHS            = 30
    TRAIN_FRACTION_A    = 0.1
    TRAIN_FRACTIONS     = [0.01, 0.02, 0.03, 0.05, 0.1]
    CNN_CHANNELS_A      = [2, 4, 8, 16, 32, 64, 128]
    CNN_CHANNELS_B      = 8
    TEST_FRACTION_CURVE = 0.1
    TEST_FRACTION_FINAL = 1.0

L             = 2
KERNEL_SIZE   = 5
BATCH_SIZE    = 128
LR            = 1e-3
N_CLASSES     = 10
N_CONV_LAYERS = 2
CONV_PR_POOL  = 1
NUM_WORKERS   = 4

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR  = FAG_PROJEKT_DIR / "reports" / "param_datasize"
RES_DIR  = FAG_PROJEKT_DIR / "results"
RES_PATH = RES_DIR / "param_datasize_results.pt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


# Fixed test set (same eval target for every config/seed). Built first so IMG_SIZE
# can be read off the data instead of hardcoded (28 normally, 36 when padded).

test_curve_ds = RotatedMNIST(split="test", rotated=True, fraction=TEST_FRACTION_CURVE, padding=PADDING)
test_final_ds = RotatedMNIST(split="test", rotated=True, fraction=TEST_FRACTION_FINAL, padding=PADDING)
test_curve_loader = DataLoader(test_curve_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_final_loader = DataLoader(test_final_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

IMG_SIZE = test_curve_ds.images.shape[-1]

ARCH_KWARGS = dict(kernel_size=KERNEL_SIZE, img_size=IMG_SIZE, n_classes=N_CLASSES,
                   n_conv_layers=N_CONV_LAYERS, conv_pr_pool=CONV_PR_POOL)

config = dict(smoke=SMOKE, padding=PADDING, seeds=list(SEEDS), n_epochs=N_EPOCHS,
              img_size=IMG_SIZE, l=L, kernel_size=KERNEL_SIZE, batch_size=BATCH_SIZE, lr=LR,
              train_fraction_a=TRAIN_FRACTION_A, train_fractions=TRAIN_FRACTIONS,
              cnn_channels_a=CNN_CHANNELS_A, cnn_channels_b=CNN_CHANNELS_B,
              test_fraction_final=TEST_FRACTION_FINAL, device=DEVICE)

exp_a, exp_b = [], []   # filled incrementally; saved after every config


# Helpers

def count_params(model):
    model(torch.zeros(1, 1, IMG_SIZE, IMG_SIZE))   # materialise LazyLinear
    return sum(p.numel() for p in model.parameters())


def make_cnn(channels):
    return CNN(channels=channels, **ARCH_KWARGS)


def make_ge(channels, n_rings):
    return GE_CNN(l=L, channels=channels, n_rings=n_rings, **ARCH_KWARGS)


def find_matching_ge_config(target_params):
    """Largest GE-CNN with <= target_params (conservative match), tuned via n_rings.
    Params are init-independent, so this runs once per target, not per seed."""
    channel_opts = (2, 4, 8, 16, 32, 64, 128)
    nring_range  = range(2, KERNEL_SIZE + 2)      # up to 6 rings for a 5x5 kernel
    best, fallback = None, None
    for ch in channel_opts:
        for nr in nring_range:
            n = count_params(make_ge(ch, nr))
            if fallback is None or n < fallback[2]:
                fallback = (ch, nr, n)
            if n <= target_params and (best is None or target_params - n < best[0]):
                best = (target_params - n, ch, nr, n)
    return (best[1], best[2], best[3]) if best else fallback


def make_train_loader(fraction, seed):
    ds = RotatedMNIST(split="train", rotated=True, fraction=fraction, padding=PADDING, seed=seed)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, generator=g), len(ds)


def train_and_eval(build_model, train_loader, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = build_model()
    train_loop(model, lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS,
               test_loader=test_curve_loader, show_progress=False, device=DEVICE)
    _, final_acc = evaluate(model, test_final_loader, show_progress=False, device=DEVICE)
    return final_acc


def run_pair(build_ge, build_cnn, fraction):
    """Train GE and CNN over all seeds at the given data fraction; return acc lists."""
    ge_accs, cnn_accs, n_train = [], [], None
    for seed in SEEDS:
        ge_loader,  n_train = make_train_loader(fraction, seed)
        cnn_loader, _       = make_train_loader(fraction, seed)   # same subset & order
        ge_accs.append(train_and_eval(build_ge, ge_loader, seed))
        cnn_accs.append(train_and_eval(build_cnn, cnn_loader, seed))
    return ge_accs, cnn_accs, n_train


def m_s(accs):
    a = np.asarray(accs, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if a.size > 1 else 0.0)


def save_all():
    torch.save({"config": config, "exp_a": exp_a, "exp_b": exp_b}, RES_PATH)


# Run

print(f"Device: {DEVICE} | SMOKE={SMOKE} | seeds={list(SEEDS)} | epochs={N_EPOCHS} | img={IMG_SIZE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Experiment A: parameter vs parameter
print("\n==== Experiment A: parameter vs parameter ====")
for cnn_ch in CNN_CHANNELS_A:
    target = count_params(make_cnn(cnn_ch))
    ge_ch, ge_nr, ge_params = find_matching_ge_config(target)
    print(f"\nCNN ch={cnn_ch}: {target:,}p  ->  GE ch={ge_ch}, rings={ge_nr}: {ge_params:,}p")
    try:
        ge_accs, cnn_accs, _ = run_pair(lambda: make_ge(ge_ch, ge_nr),
                                        lambda: make_cnn(cnn_ch), TRAIN_FRACTION_A)
        ge_m, ge_s = m_s(ge_accs); cnn_m, cnn_s = m_s(cnn_accs)
        print(f"  GE {ge_m:.3f}+/-{ge_s:.3f} | CNN {cnn_m:.3f}+/-{cnn_s:.3f}")
        exp_a.append({"cnn_channels": cnn_ch, "cnn_params": target,
                      "ge_channels": ge_ch, "ge_n_rings": ge_nr, "ge_params": ge_params,
                      "ge_accs": ge_accs, "cnn_accs": cnn_accs})
        save_all()
    except RuntimeError as e:
        print(f"  SKIPPED cnn_ch={cnn_ch}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Experiment B: data efficiency (matched pair)
print("\n==== Experiment B: data efficiency ====")
target_b = count_params(make_cnn(CNN_CHANNELS_B))
ge_ch_b, ge_nr_b, ge_params_b = find_matching_ge_config(target_b)
print(f"Matched pair: CNN ch={CNN_CHANNELS_B} ({target_b:,}p) ~ GE ch={ge_ch_b}, rings={ge_nr_b} ({ge_params_b:,}p)")
for frac in TRAIN_FRACTIONS:
    try:
        ge_accs, cnn_accs, n_train = run_pair(lambda: make_ge(ge_ch_b, ge_nr_b),
                                              lambda: make_cnn(CNN_CHANNELS_B), frac)
        ge_m, _ = m_s(ge_accs); cnn_m, _ = m_s(cnn_accs)
        print(f"  frac {frac:.0%} (n={n_train}): GE {ge_m:.3f} | CNN {cnn_m:.3f}")
        exp_b.append({"fraction": frac, "n_train": n_train,
                      "cnn_params": target_b, "ge_params": ge_params_b,
                      "ge_accs": ge_accs, "cnn_accs": cnn_accs})
        save_all()
    except RuntimeError as e:
        print(f"  SKIPPED frac={frac}: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# JSON summary
summary = {
    "config": config,
    "exp_a": [{"cnn_params": r["cnn_params"], "ge_params": r["ge_params"],
               "ge": dict(zip(("mean", "std"), m_s(r["ge_accs"]))),
               "cnn": dict(zip(("mean", "std"), m_s(r["cnn_accs"])))} for r in exp_a],
    "exp_b": [{"fraction": r["fraction"], "n_train": r["n_train"],
               "ge": dict(zip(("mean", "std"), m_s(r["ge_accs"]))),
               "cnn": dict(zip(("mean", "std"), m_s(r["cnn_accs"])))} for r in exp_b],
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)


# Plots (mean + std error bars)

if exp_a:
    rs = sorted(exp_a, key=lambda r: r["cnn_params"])
    cnn_p = [r["cnn_params"] for r in rs]; ge_p = [r["ge_params"] for r in rs]
    ge_m  = [m_s(r["ge_accs"])[0]  for r in rs]; ge_s  = [m_s(r["ge_accs"])[1]  for r in rs]
    cnn_m = [m_s(r["cnn_accs"])[0] for r in rs]; cnn_s = [m_s(r["cnn_accs"])[1] for r in rs]
    adv   = [g - c for g, c in zip(ge_m, cnn_m)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(ge_p,  ge_m,  yerr=ge_s,  fmt="o-", capsize=4, label="GE-CNN", color="tab:blue")
    ax1.errorbar(cnn_p, cnn_m, yerr=cnn_s, fmt="s-", capsize=4, label="CNN",    color="tab:orange")
    ax1.set_xscale("log"); ax1.set_xlabel("Parameters (log)"); ax1.set_ylabel("Test accuracy (rotated)")
    ax1.set_title("Accuracy vs parameter count"); ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.axhline(0, color="black", lw=1)
    ax2.plot(cnn_p, adv, "o-", color="tab:green")
    ax2.set_xscale("log"); ax2.set_xlabel("Parameters (log)")
    ax2.set_ylabel("GE-CNN - CNN accuracy"); ax2.set_title("Equivariance advantage vs size")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Experiment A: param vs param (mean +/- std over {len(list(SEEDS))} seeds)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "param_vs_param.png", dpi=150); plt.close(fig)

if exp_b:
    rs = sorted(exp_b, key=lambda r: r["n_train"])
    n_train = [r["n_train"] for r in rs]
    ge_m  = [m_s(r["ge_accs"])[0]  for r in rs]; ge_s  = [m_s(r["ge_accs"])[1]  for r in rs]
    cnn_m = [m_s(r["cnn_accs"])[0] for r in rs]; cnn_s = [m_s(r["cnn_accs"])[1] for r in rs]
    adv   = [g - c for g, c in zip(ge_m, cnn_m)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(n_train, ge_m,  yerr=ge_s,  fmt="o-", capsize=4, label="GE-CNN", color="tab:blue")
    ax1.errorbar(n_train, cnn_m, yerr=cnn_s, fmt="s-", capsize=4, label="CNN",    color="tab:orange")
    ax1.set_xscale("log"); ax1.set_xlabel("Training images (log)"); ax1.set_ylabel("Test accuracy (rotated)")
    ax1.set_title("Accuracy vs data size"); ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.axhline(0, color="black", lw=1)
    ax2.plot(n_train, adv, "o-", color="tab:green")
    ax2.fill_between(n_train, adv, 0, where=[a >= 0 for a in adv], alpha=0.2, color="tab:green")
    ax2.fill_between(n_train, adv, 0, where=[a < 0 for a in adv],  alpha=0.2, color="tab:red")
    ax2.set_xscale("log"); ax2.set_xlabel("Training images (log)")
    ax2.set_ylabel("GE-CNN - CNN accuracy"); ax2.set_title("Equivariance advantage vs data")
    ax2.grid(True, alpha=0.3)
    fig.suptitle(f"Experiment B: data efficiency (mean +/- std over {len(list(SEEDS))} seeds)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "data_efficiency.png", dpi=150); plt.close(fig)

print(f"\nDone. Results: {RES_PATH}")
print(f"Plots + summary: {OUT_DIR}")