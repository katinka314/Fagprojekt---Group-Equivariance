from pathlib import Path
import random

import numpy as np
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "archive"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


class RotatedMNIST(Dataset):
    """Rotated MNIST dataset loaded from preprocessed .pt files.

    fraction: share of the dataset to use, e.g. 0.1 for 10 %. Selected as a
    random but reproducible subset (controlled by seed).
    stratified: if True, the same fraction is drawn from each class, so the
    label distribution is preserved exactly in the subset.
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_OUTPUT_DIR,
        split: str = "train",
        fraction: float = 1.0,
        stratified: bool = True,
        seed: int = 42,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1], got {fraction!r}")

        data_dir = Path(data_dir)
        self.images = torch.load(data_dir / f"rotated_{split}_images.pt")
        self.labels = torch.load(data_dir / f"rotated_{split}_labels.pt")
        self.angles = torch.load(data_dir / f"rotated_{split}_angles.pt")

        if fraction < 1.0:
            generator = torch.Generator().manual_seed(seed)
            if stratified:
                per_label_indices = []
                for label in self.labels.unique():
                    label_idx = (self.labels == label).nonzero(as_tuple=True)[0]
                    n_label = int(len(label_idx) * fraction)
                    perm = torch.randperm(len(label_idx), generator=generator)[:n_label]
                    per_label_indices.append(label_idx[perm])
                indices = torch.cat(per_label_indices)
                # Shuffle at the end so the subset is not ordered by class.
                indices = indices[torch.randperm(len(indices), generator=generator)]
            else:
                n_samples = int(len(self.images) * fraction)
                indices = torch.randperm(len(self.images), generator=generator)[:n_samples]
            self.images = self.images[indices]
            self.labels = self.labels[indices]
            self.angles = self.angles[indices]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # uint8 [28, 28] -> float32 [1, 28, 28] in [0, 1]
        image = self.images[idx].unsqueeze(0).float() / 255.0
        return image, self.labels[idx], self.angles[idx]


def load_mnist_images(path: Path) -> torch.Tensor:
    """Load MNIST images from an IDX file."""
    with open(path, "rb") as f:
        data = f.read()

    images = np.frombuffer(data, dtype=np.uint8, offset=16).copy()
    images = torch.from_numpy(images).reshape(-1, 28, 28)

    return images


def load_mnist_labels(path: Path) -> torch.Tensor:
    """Load MNIST labels from an IDX file."""
    with open(path, "rb") as f:
        data = f.read()

    labels = np.frombuffer(data, dtype=np.uint8, offset=8).copy()
    labels = torch.from_numpy(labels)

    return labels


def rotate_dataset(images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate each image by a random angle in [0, 360)."""
    rotated_images = []
    angles = []

    for image in images:
        angle = random.uniform(0, 360)

        pil_image = Image.fromarray(image.numpy(), mode="L")
        rotated = rotate(pil_image, angle=angle, fill=0)

        rotated_array = np.array(rotated, dtype=np.uint8)
        rotated_tensor = torch.from_numpy(rotated_array)

        rotated_images.append(rotated_tensor)
        angles.append(angle)

    rotated_images = torch.stack(rotated_images)
    angles = torch.tensor(angles, dtype=torch.float32)

    return rotated_images, angles


def preprocess_split(
    image_path: Path,
    label_path: Path,
    output_image_path: Path,
    output_label_path: Path,
    output_angle_path: Path,
) -> None:
    """Load one split, rotate all images randomly, and save images, labels, and angles."""
    print(f"Loading images from {image_path}")
    images = load_mnist_images(image_path)

    print(f"Loading labels from {label_path}")
    labels = load_mnist_labels(label_path)

    print("Rotating images with one random angle per image")
    rotated_images, angles = rotate_dataset(images)

    output_image_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving rotated images to {output_image_path}")
    torch.save(rotated_images, output_image_path)

    print(f"Saving labels to {output_label_path}")
    torch.save(labels, output_label_path)

    print(f"Saving rotation angles to {output_angle_path}")
    torch.save(angles, output_angle_path)


def preprocess(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
) -> None:
    """Create a fixed rotated MNIST dataset and save it to disk."""
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using data directory: {data_dir}")
    print(f"Using output directory: {output_dir}")

    preprocess_split(
        image_path=data_dir / "train-images.idx3-ubyte",
        label_path=data_dir / "train-labels.idx1-ubyte",
        output_image_path=output_dir / "rotated_train_images.pt",
        output_label_path=output_dir / "rotated_train_labels.pt",
        output_angle_path=output_dir / "rotated_train_angles.pt",
    )

    preprocess_split(
        image_path=data_dir / "t10k-images.idx3-ubyte",
        label_path=data_dir / "t10k-labels.idx1-ubyte",
        output_image_path=output_dir / "rotated_test_images.pt",
        output_label_path=output_dir / "rotated_test_labels.pt",
        output_angle_path=output_dir / "rotated_test_angles.pt",
    )


if __name__ == "__main__":
    typer.run(preprocess)