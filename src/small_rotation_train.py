"""Small local training run that produces the CNN + GE-CNN weights used by the
equivariance-quantify experiment.

Trains on UNROTATED, padded Fashion-MNIST and reports a rotated-test accuracy as a
sanity check. The model-argument dicts (CNN_ARGS / GE_ARGS) are defined at module
level so the quantify script can import them directly ("model params from here"),
or read them back from each checkpoint's `model_args`.

Run from the project root:   uv run python src/small_rotation_train.py
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

FAG_PROJEKT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FAG_PROJEKT_DIR))

from data import RotatedMNIST, PADDING as PAD_PX
from train import train_loop, evaluate
from models.Model import CNN, GE_CNN


# ---- Config (the single source of truth for the small run) ----
FRACTION   = 0.03      # train on 3% of the data
N_EPOCHS   = 6
TEST_FRAC  = 0.1       # rotated-test sanity on 10%
PADDING    = True
CHANNELS   = 4
L          = 2
N_RINGS    = 6
KERNEL     = 5
N_CONV     = 2
CONV_PP    = 1
N_CLASSES  = 10
LR         = 1e-3
BATCH      = 128
SEED       = 0


DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 28 + 2 * PAD_PX if PADDING else 28
SAVE_DIR = FAG_PROJEKT_DIR / "models" / "model_weights"

# Model arguments — imported by the quantify script.
_COMMON  = dict(kernel_size=KERNEL, in_features=1, img_size=IMG_SIZE, n_conv_layers=N_CONV,
                conv_pr_pool=CONV_PP, channels=CHANNELS, n_classes=N_CLASSES, bias=True)
CNN_ARGS = dict(_COMMON)
GE_ARGS  = dict(_COMMON, l=L, n_rings=N_RINGS)

MODELS = [("CNN", CNN, CNN_ARGS), ("GE_CNN", GE_CNN, GE_ARGS)]


def weight_path(name):
    return SAVE_DIR / f"{name}_smallrot_ch{CHANNELS}_l{L}_r{N_RINGS}_3pct.pth"


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)

    train_ds = RotatedMNIST(split="train", rotated=False, padding=PADDING, fraction=FRACTION, seed=SEED)
    test_ds  = RotatedMNIST(split="test",  rotated=True,  padding=PADDING, fraction=TEST_FRAC, seed=SEED)
    print(f"device={DEVICE} img_size={IMG_SIZE} n_train={len(train_ds)} n_test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    for name, Model, args in MODELS:
        print(f"\n=== training {name} (ch={CHANNELS}) ===")
        model = Model(**args)
        model.name = name
        train_loop(model, lr=LR, train_loader=train_loader, n_epochs=N_EPOCHS,
                   test_loader=test_loader, show_progress=False, device=DEVICE)
        _, acc = evaluate(model, test_loader, show_progress=False, device=DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        out = weight_path(name)
        torch.save({"model_name": name, "model_args": args, "state_dict": model.state_dict()}, out)
        print(f"{name}: rot{int(TEST_FRAC*100)}%-test acc={acc:.4f}  params={n_params:,}  saved -> {out.name}")

    print("\nDONE")


if __name__ == "__main__":
    main()
