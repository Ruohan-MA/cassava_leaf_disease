from dataclasses import dataclass, field
from pathlib import Path
import torch


BACKBONE_FEATURE_DIM = {
    "resnet34": 512,
    "resnet50": 2048,
    "efficientnet_b4": 1792,
    "densenet121": 1024,
}


class ConfigError(Exception):
    pass


@dataclass
class Config:
    # Paths
    data_dir: str = "."
    train_csv: str = "train.csv"
    train_images_dir: str = "train_images"
    test_images_dir: str = "test_images"
    sample_submission: str = "sample_submission.csv"
    checkpoint_dir: str = "checkpoints"
    split_file: str = "splits/train_val_split.csv"
    metrics_file: str = "checkpoints/baseline_metrics.csv"
    ssl_metrics_file: str = "checkpoints/ssl_metrics.csv"
    finetune_metrics_file: str = "checkpoints/finetune_metrics.csv"

    # Backbone
    backbone: str = "resnet34"

    # Reproducibility
    seed: int = 42

    # Data split
    val_split: float = 0.2

    # Image
    img_size: int = 224

    # DataLoader
    batch_size: int = 256
    num_workers: int = 4

    # Supervised baseline training
    lr: float = 1e-3
    epochs: int = 30
    weight_decay: float = 1e-4

    # BYOL SSL pretraining
    ssl_lr: float = 3e-4
    ssl_epochs: int = 100
    ssl_weight_decay: float = 1e-4
    ema_momentum: float = 0.996
    projection_dim: int = 256
    hidden_dim: int = 4096
    save_every: int = 10
    byol_collapse_std_threshold: float = 0.01

    # SSL augmentation
    ssl_crop_scale_min: float = 0.4
    ssl_color_jitter_strength: float = 0.5

    # Fine-tuning
    finetune_lr_backbone: float = 1e-4
    finetune_lr_head: float = 1e-3
    finetune_epochs: int = 30
    finetune_weight_decay: float = 1e-4

    # Class imbalance strategy (exactly one must be True)
    use_weighted_ce: bool = True
    use_balanced_sampler: bool = False

    # ImageNet normalization constants
    imagenet_mean: tuple = (0.485, 0.456, 0.406)
    imagenet_std: tuple = (0.229, 0.224, 0.225)

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def __post_init__(self):
        if self.use_weighted_ce and self.use_balanced_sampler:
            raise ConfigError(
                "use_weighted_ce and use_balanced_sampler cannot both be True. "
                "Choose exactly one class imbalance strategy."
            )
        if self.backbone not in BACKBONE_FEATURE_DIM:
            raise KeyError(
                f"Unsupported backbone '{self.backbone}'. "
                f"Choose from: {list(BACKBONE_FEATURE_DIM.keys())}"
            )

    @property
    def feature_dim(self) -> int:
        return BACKBONE_FEATURE_DIM[self.backbone]

    @property
    def train_images_path(self) -> Path:
        return Path(self.data_dir) / self.train_images_dir

    @property
    def test_images_path(self) -> Path:
        return Path(self.data_dir) / self.test_images_dir

    @property
    def train_csv_path(self) -> Path:
        return Path(self.data_dir) / self.train_csv

    @property
    def split_file_path(self) -> Path:
        return Path(self.data_dir) / self.split_file

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.data_dir) / self.checkpoint_dir


NUM_CLASSES = 5
DISEASE_MAP = {0: "CBB", 1: "CBSD", 2: "CGM", 3: "CMD", 4: "Healthy"}
