"""
Ensemble inference: load multiple trained checkpoints (any backbone mix),
average their softmax probabilities, and write a Kaggle submission CSV.

Usage:
    python ensemble_inference.py \
        --model checkpoints/resnet34_best.pt resnet34 \
        --model checkpoints/efficientnet_b4_best.pt efficientnet_b4 \
        --model checkpoints/densenet121_best.pt densenet121 \
        --output submission.csv

    With TTA (5 augmentation rounds per model):
        python ensemble_inference.py \
            --model checkpoints/resnet34_best.pt resnet34 \
            --model checkpoints/efficientnet_b4_best.pt efficientnet_b4 \
            --tta 5 --output submission.csv
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from config import Config, NUM_CLASSES, BACKBONE_FEATURE_DIM
from augmentations import build_val_transform, build_train_transform
from model_utils import LeafClassifier, build_backbone
from inference import TestDataset


def _load_model(checkpoint_path: Path, backbone_name: str, device: torch.device) -> LeafClassifier:
    backbone = build_backbone(backbone_name, pretrained=False)
    model = LeafClassifier(backbone, BACKBONE_FEATURE_DIM[backbone_name], NUM_CLASSES)

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    n_classes = state_dict["head.fc.weight"].shape[0]
    if n_classes != NUM_CLASSES:
        raise RuntimeError(
            f"{checkpoint_path}: checkpoint has {n_classes} classes, expected {NUM_CLASSES}."
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _run_pass(model, loader, device, n_samples):
    """Single forward pass over the test set; returns (n_samples, NUM_CLASSES) prob tensor."""
    probs = torch.zeros(n_samples, NUM_CLASSES)
    offset = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_probs = F.softmax(model(images), dim=1).cpu()
            probs[offset : offset + batch_probs.size(0)] = batch_probs
            offset += batch_probs.size(0)
    return probs


def ensemble_infer(
    model_specs: list[tuple[str, str]],
    cfg: Config = None,
    output_csv: str | Path = "submission.csv",
    tta_n: int = 1,
):
    """
    Ensemble inference across multiple models with optional TTA.

    Args:
        model_specs: list of (checkpoint_path, backbone_name) pairs.
        cfg: Config object. Defaults to Config() if None.
        output_csv: path for the output submission CSV.
        tta_n: augmentation rounds per model.
                1 = single deterministic pass (val transform).
                >1 = tta_n random-augmentation passes (train transform) per model.
    """
    if cfg is None:
        cfg = Config()

    device = torch.device(cfg.device)

    submission_path = Path(cfg.data_dir) / cfg.sample_submission
    if not submission_path.exists():
        raise FileNotFoundError(f"sample_submission.csv not found at {submission_path}")

    test_dir = cfg.test_images_path
    if not test_dir.exists():
        raise FileNotFoundError(f"test_images directory not found: {test_dir}")

    image_ids = pd.read_csv(submission_path)["image_id"].tolist()
    n_samples = len(image_ids)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_transform = build_val_transform(cfg)
    tta_transform = build_train_transform(cfg) if tta_n > 1 else None

    accumulated = torch.zeros(n_samples, NUM_CLASSES)
    total_passes = len(model_specs) * tta_n

    for checkpoint_path, backbone_name in model_specs:
        print(f"Loading {backbone_name} from {checkpoint_path} ...")
        model = _load_model(Path(checkpoint_path), backbone_name, device)

        for r in range(tta_n):
            transform = val_transform if tta_n == 1 else tta_transform
            tag = "" if tta_n == 1 else f"  TTA round {r + 1}/{tta_n}"
            if tag:
                print(tag)
            loader = DataLoader(
                TestDataset(image_ids, test_dir, transform=transform),
                **loader_kwargs,
            )
            accumulated += _run_pass(model, loader, device, n_samples)

    preds = (accumulated / total_passes).argmax(dim=1).tolist()
    pd.DataFrame({"image_id": image_ids, "label": preds}).to_csv(output_csv, index=False)
    print(
        f"Ensemble submission written to {output_csv} "
        f"({n_samples} rows, {len(model_specs)} model(s), TTA={tta_n})."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble inference for cassava leaf disease classification."
    )
    parser.add_argument(
        "--model",
        nargs=2,
        metavar=("CHECKPOINT", "BACKBONE"),
        action="append",
        required=True,
        help=(
            "Checkpoint path and backbone name. Repeat for each model. "
            f"Supported: {list(BACKBONE_FEATURE_DIM.keys())}"
        ),
    )
    parser.add_argument("--output", default="submission.csv", help="Output CSV path")
    parser.add_argument("--data-dir", default=".", help="Root data directory")
    parser.add_argument(
        "--tta",
        type=int,
        default=1,
        help="TTA rounds per model (default 1 = no TTA, uses val transform)",
    )
    args = parser.parse_args()

    cfg = Config(data_dir=args.data_dir)
    ensemble_infer(
        model_specs=[(ckpt, backbone) for ckpt, backbone in args.model],
        cfg=cfg,
        output_csv=args.output,
        tta_n=args.tta,
    )
