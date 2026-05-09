"""
Inference script: load a trained classifier checkpoint and produce a Kaggle submission CSV.

Reads all images from test_images/, runs forward passes, and writes image_id + label columns
matching the sample_submission.csv format.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from config import Config, NUM_CLASSES
from augmentations import build_val_transform
from model_utils import LeafClassifier, build_backbone


class TestDataset(Dataset):
    """Loads test images in a deterministic order derived from sample_submission.csv."""

    def __init__(self, image_ids: list[str], images_dir: Path, transform=None):
        self.image_ids = image_ids
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        image_id = self.image_ids[idx]
        img_path = self.images_dir / image_id
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, image_id


def infer(
    checkpoint_path: str | Path,
    cfg: Config = None,
    output_csv: str | Path = "submission.csv",
):
    """
    Run inference on test_images/ and write a Kaggle submission CSV.

    Args:
        checkpoint_path: path to a full model state_dict (LeafClassifier).
        cfg: Config object. Defaults to Config() if None.
        output_csv: path for the output submission CSV.
    """
    if cfg is None:
        cfg = Config()

    checkpoint_path = Path(checkpoint_path)
    output_csv = Path(output_csv)
    device = torch.device(cfg.device)

    submission_path = Path(cfg.data_dir) / cfg.sample_submission
    if not submission_path.exists():
        raise FileNotFoundError(f"sample_submission.csv not found at {submission_path}")

    sample_df = pd.read_csv(submission_path)
    image_ids = sample_df["image_id"].tolist()

    test_dir = cfg.test_images_path
    if not test_dir.exists():
        raise FileNotFoundError(f"test_images directory not found: {test_dir}")

    transform = build_val_transform(cfg)
    test_dataset = TestDataset(image_ids, test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    backbone = build_backbone(cfg.backbone, pretrained=False)
    model = LeafClassifier(backbone, cfg.feature_dim, NUM_CLASSES)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    num_classes_in_ckpt = state_dict["head.fc.weight"].shape[0]
    if num_classes_in_ckpt != NUM_CLASSES:
        raise RuntimeError(
            f"Checkpoint was trained for {num_classes_in_ckpt} classes, "
            f"but NUM_CLASSES={NUM_CLASSES}. Wrong checkpoint for this task."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_ids.extend(ids)

    output_df = pd.DataFrame({"image_id": all_ids, "label": all_preds})
    output_df.to_csv(output_csv, index=False)
    print(f"Submission written to {output_csv} ({len(output_df)} rows).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--output", default="submission.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--data-dir", default=".", help="Root data directory"
    )
    args = parser.parse_args()

    cfg = Config(data_dir=args.data_dir)
    infer(
        checkpoint_path=args.checkpoint,
        cfg=cfg,
        output_csv=args.output,
    )
