"""
Laver et roteret Fashion-MNIST datasaet.

Metode: for hvert billede i train og test traekkes en tilfaeldig vinkel
uniformt i [0, 360) grader, og billedet roteres med den vinkel.

Output (matcher det eksisterende format i data/processed/):
    rotated_{split}_images.pt   (N, 28, 28)  uint8
    rotated_{split}_labels.pt   (N,)         uint8
    rotated_{split}_angles.pt   (N,)         float32   (grader, CCW)

Koer:
    python notebooks/Rot_Mnist_creator.py
"""

from __future__ import annotations

from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torchvision.transforms.functional import rotate, InterpolationMode
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
SEED = 0
KAGGLE_DATASET = "zalando-research/fashionmnist"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
IMG_SIZE = 28
INTERPOLATION = InterpolationMode.BILINEAR  # BILINEAR = glat, NEAREST = ingen blur men aliasing
FILL = 0  # baggrundsvaerdi; MNIST-baggrund er sort, saa 0 passer


def get_csv_dir() -> Path:
    """
    Henter Fashion-MNIST via kagglehub og returnerer mappen med csv-filerne.

    kagglehub downloader til den lokale brugers eget cache (og springer over hvis
    det allerede er hentet), saa stien virker for alle i gruppen ud fra repoet -
    ingen maskine-specifik sti haardkodet.
    """
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def load_fashion_mnist(csv_path: Path) -> tuple[Tensor, Tensor]:
    """Laeser et fashion-mnist csv og returnerer (images uint8 (N,28,28), labels uint8 (N,))."""
    df = pd.read_csv(csv_path)
    labels = torch.tensor(df.iloc[:, 0].to_numpy(), dtype=torch.uint8)
    pixels = df.iloc[:, 1:].to_numpy().reshape(-1, IMG_SIZE, IMG_SIZE).astype(np.uint8)
    images = torch.from_numpy(pixels)  # (N, 28, 28) uint8
    return images, labels


def rotate_dataset(images: Tensor, generator: torch.Generator, desc: str = "rotating",) -> tuple[Tensor, Tensor]:
    """
    Roterer hvert billede med en uniform tilfaeldig vinkel i [0, 360).

    Returnerer (roterede billeder uint8 (N,28,28), vinkler float32 (N,) i grader).
    """
    n = images.shape[0]
    angles = torch.rand(n, generator=generator) * 360.0  # [0, 360), uniform

    rotated = torch.empty_like(images)
    for i in tqdm(range(n), desc=desc, unit="img"):
        # rotate forventer (C, H, W); vi tilfoejer og fjerner kanal-aksen igen.
        img = images[i].unsqueeze(0)  # (1, 28, 28) uint8
        rot = rotate(img, angle=float(angles[i]), interpolation=INTERPOLATION,
                     expand=False,        # behold 28x28 (hjoerner kan klippes - se noter)
                    fill=FILL)
        rotated[i] = rot.squeeze(0)

    return rotated, angles.to(torch.float32)


def save_split(split: str, images: Tensor, labels: Tensor, angles: Tensor) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(images, OUT_DIR / f"rotated_{split}_images.pt")
    torch.save(labels, OUT_DIR / f"rotated_{split}_labels.pt")
    torch.save(angles, OUT_DIR / f"rotated_{split}_angles.pt")
    print(f"[{split}] gemt: {images.shape} billeder, vinkler {angles.min():.1f}-{angles.max():.1f} grader")


def sanity_check(split: str = "train", n: int = 8, seed: int = np.random.randint) -> None:
    """Viser n tilfaeldige roterede billeder med deres label og vinkel."""
    import matplotlib.pyplot as plt

    images = torch.load(OUT_DIR / f"rotated_{split}_images.pt")
    labels = torch.load(OUT_DIR / f"rotated_{split}_labels.pt")
    angles = torch.load(OUT_DIR / f"rotated_{split}_angles.pt")

    rng = np.random.default_rng(seed)
    idx = rng.choice(images.shape[0], size=n, replace=False)

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.5))
    for ax, i in zip(axes, idx):
        ax.imshow(images[i].numpy(), cmap="gray")
        ax.set_title(f"y={int(labels[i])}\n{float(angles[i]):.0f} deg")
        ax.axis("off")
    fig.suptitle(f"Roteret Fashion-MNIST ({split})")
    plt.tight_layout()
    plt.show()


def split_exists(split: str) -> bool:
    """True hvis alle tre .pt-filer for et split allerede findes."""
    return all((OUT_DIR / f"rotated_{split}_{k}.pt").exists() for k in ("images", "labels", "angles"))


def main(force: bool = False) -> None:
    gen = torch.Generator().manual_seed(SEED)
    csv_dir: Path | None = None  # hentes foerst naar et split faktisk skal genereres

    for split, csv_name in [("train", "fashion-mnist_train.csv"), ("test", "fashion-mnist_test.csv")]:
        if split_exists(split) and not force:
            print(f"[{split}] findes allerede, springer over (koer med --force for at regenerere)")
            continue
        if csv_dir is None:
            csv_dir = get_csv_dir()
        images, labels = load_fashion_mnist(csv_dir / csv_name)
        rotated, angles = rotate_dataset(images, gen, desc=f"rotating {split}")
        save_split(split, rotated, labels, angles)

    sanity_check("train")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lav roteret Fashion-MNIST datasaet.")
    parser.add_argument("--force", action="store_true", help="Regenerer selvom .pt-filerne allerede findes")
    args = parser.parse_args()
    main(force=args.force)
