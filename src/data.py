from pathlib import Path
import random

import numpy as np
import torch
import typer
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate, InterpolationMode
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "archive"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
PADDING = 4

class RotatedMNIST(Dataset):
    """Rotated MNIST dataset loaded from preprocessed .pt files.
    Normally: Just specify split, and rotated
    
    Parameters 
    ----------
    Returns: 
        image, self.labels[idx], self.angles[idx]
    split: str
        "train" "test"
    rotated:
        True False
  
    stratified: 
        True False
    Ex:
    train_set = RotatedMNIST(split="train", rotated=True)  
        (train on rotated)
    test_set  = RotatedMNIST(split="test",  rotated=False)  
        (test on normal)
    """

    def __init__(self, data_dir: Path = DEFAULT_OUTPUT_DIR, split: str = "train", rotated: bool = True, fraction: float = 1.0, stratified: bool = True, seed: int = 42) -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test'")
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"fraction must be in (0, 1]")
        prefix = 'rotated_' if rotated else ''

        data_dir = Path(data_dir)
        self.images = torch.load(data_dir / f"{prefix}{split}_images.pt")
        self.labels = torch.load(data_dir / f"{split}_labels.pt")
        if rotated:
            self.angles = torch.load(data_dir / f"{prefix}{split}_angles.pt")
        else:
            self.angles = torch.zeros(len(self.images), dtype= torch.float32)

        if fraction < 1.0:
            generator = torch.Generator().manual_seed(seed)
            if stratified:
                per_label_indices = []
                #for label in self.labels.unique():
                #print(self.labels.unique())
                for label in sorted(set(self.labels.tolist())):
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

    images = np.frombuffer(data, dtype=np.uint8, offset=16).copy() #Returns images as 1D arrays of len N*784
    images = torch.from_numpy(images).reshape(-1, 28, 28) # Transforms into torch 28x 28 images 

    return images # (i x r x c) #range 0, 255

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
        rotated = rotate(pil_image, angle=angle, fill=0, interpolation = InterpolationMode.BILINEAR)

        rotated_array = np.array(rotated, dtype=np.uint8)
        rotated_tensor = torch.from_numpy(rotated_array)

        rotated_images.append(rotated_tensor)
        angles.append(angle)

    rotated_images = torch.stack(rotated_images)
    angles = torch.tensor(angles, dtype=torch.float32)

    return rotated_images, angles


def preprocess_split(image_path: Path, label_path: Path, output_dir: Path, split: str, pad = PADDING) -> None:
    """Load one split, save plain images, rotated images, angles, and shared labels."""

    images = load_mnist_images(image_path)
    labels = load_mnist_labels(label_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    padded_images = F.pad(images, (pad,pad,pad,pad), value = 0)
    torch.save(images, output_dir / f"{split}_images.pt")
    torch.save(images, output_dir / f"{split}_images_padded.pt")
    rotated_images, angles = rotate_dataset(images)
    torch.save(rotated_images, output_dir / f"rotated_{split}_images.pt")
    torch.save(angles, output_dir / f"rotated_{split}_angles.pt")
    torch.save(labels, output_dir / f"{split}_labels.pt")
    rotated_images, angles = rotate_dataset(padded_images)
    torch.save(rotated_images, output_dir / f"rotated_{split}_images_padded.pt")
    torch.save(angles, output_dir / f"rotated_{split}_angles_padded.pt")
    torch.save(labels, output_dir / f"{split}_labels_padded.pt")

    print('All images are proccessed')



def preprocess(data_dir: Path = DEFAULT_DATA_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR, seed: int = 42) -> None:
    """Create a fixed and, rotated MNIST dataset and save it to disk."""
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Using data directory: {data_dir}")
    print(f"Using output directory: {output_dir}")

    preprocess_split(
    image_path=data_dir / "train-images.idx3-ubyte",
    label_path=data_dir / "train-labels.idx1-ubyte",
    output_dir=output_dir,
    split="train",
    )

    preprocess_split(
        image_path=data_dir / "t10k-images.idx3-ubyte",
        label_path=data_dir / "t10k-labels.idx1-ubyte",
        output_dir=output_dir,
        split="test",
    )


if __name__ == "__main__":
    typer.run(preprocess)