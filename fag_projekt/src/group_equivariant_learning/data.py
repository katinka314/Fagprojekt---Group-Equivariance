from pathlib import Path
import random

import numpy as np
import torch
import typer
from PIL import Image
from torchvision.transforms.functional import rotate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "archive"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


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