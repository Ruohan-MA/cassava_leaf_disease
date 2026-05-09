import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

from config import Config, ConfigError, NUM_CLASSES


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_splits(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create or reload a stratified train/val split.

    If the split CSV already exists it is reloaded unchanged.
    Otherwise a new split is generated with cfg.seed, saved, and returned.
    Returns (train_df, val_df) each with columns [image_id, label, split_path].
    """
    split_path = cfg.split_file_path
    train_csv_path = cfg.train_csv_path

    if not train_csv_path.exists():
        raise FileNotFoundError(f"train.csv not found at {train_csv_path}")

    full_df = pd.read_csv(train_csv_path)

    if split_path.exists():
        split_df = pd.read_csv(split_path)
        train_df = split_df[split_df["split"] == "train"].drop(columns=["split"])
        val_df = split_df[split_df["split"] == "val"].drop(columns=["split"])
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=cfg.val_split, random_state=cfg.seed
    )
    train_idx, val_idx = next(splitter.split(full_df, full_df["label"]))

    train_df = full_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = full_df.iloc[val_idx].copy().reset_index(drop=True)

    split_path.parent.mkdir(parents=True, exist_ok=True)

    train_tagged = train_df.copy()
    train_tagged["split"] = "train"
    val_tagged = val_df.copy()
    val_tagged["split"] = "val"
    pd.concat([train_tagged, val_tagged]).to_csv(split_path, index=False)

    return train_df, val_df


def compute_class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from training split labels only."""
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


class BalancedBatchSampler:
    """
    Yields batches where class counts differ by at most 1.

    Each batch contains floor(batch_size / n_classes) samples from each class,
    with the remainder distributed across the first (batch_size % n_classes) classes.
    Per-class pools are re-shuffled independently when exhausted.
    """

    def __init__(self, labels: np.ndarray, n_classes: int, batch_size: int):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.class_indices = [
            np.where(labels == c)[0].tolist() for c in range(n_classes)
        ]
        self._n_batches = sum(len(p) for p in self.class_indices) // batch_size

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        pools = [pool.copy() for pool in self.class_indices]
        for pool in pools:
            random.shuffle(pool)
        pos = [0] * self.n_classes
        base = self.batch_size // self.n_classes
        remainder = self.batch_size % self.n_classes

        for _ in range(self._n_batches):
            batch = []
            for c in range(self.n_classes):
                count = base + (1 if c < remainder else 0)
                for _ in range(count):
                    if pos[c] >= len(pools[c]):
                        random.shuffle(pools[c])
                        pos[c] = 0
                    batch.append(pools[c][pos[c]])
                    pos[c] += 1
            random.shuffle(batch)
            yield batch


class LeafDataset(Dataset):
    """Loads cassava leaf images with labels for supervised training/validation."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, transform=None):
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.images_dir / row["image_id"]
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class LeafSSLDataset(Dataset):
    """
    Loads cassava leaf images for BYOL pretraining.

    Returns two independently augmented views of each image.
    Only accepts paths from the training image directory — never test images.
    """

    def __init__(self, image_paths: list[Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            view1 = self.transform(img)
            view2 = self.transform(img)
        else:
            import torchvision.transforms as T
            t = T.ToTensor()
            view1 = t(img)
            view2 = t(img)
        return view1, view2


def get_ssl_image_paths(cfg: Config) -> list[Path]:
    """
    Collect image paths for BYOL pretraining from train_images/ only.

    Raises ConfigError if the configured train_images_dir matches the test directory
    or if it does not exist.
    """
    train_dir = cfg.train_images_path
    test_dir = cfg.test_images_path

    if not train_dir.exists():
        raise FileNotFoundError(f"train_images directory not found: {train_dir}")

    if train_dir.resolve() == test_dir.resolve():
        raise ConfigError(
            "BYOL pretraining must use train_images/ only — "
            "test_images/ path matches train_images/ path, which would cause data leakage."
        )

    paths = sorted(train_dir.glob("*.jpg"))
    if not paths:
        paths = sorted(train_dir.glob("*.jpeg"))
    return paths
