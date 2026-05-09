"""
SSL fine-tuning: load BYOL backbone checkpoint and fine-tune for cassava leaf classification.

Two modes:
  - linear_probe: backbone is frozen, only the classification head is trained
  - full_finetune: all layers are trained with differential LR (lower for backbone)

Class weights are computed from the training split labels only.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, ConfigError, NUM_CLASSES
from augmentations import build_train_transform, build_val_transform
from data_utils import (
    LeafDataset,
    compute_class_weights,
    make_balanced_sampler,
    make_splits,
    set_seeds,
)
from model_utils import LeafClassifier, load_backbone_from_checkpoint
from baseline import evaluate, write_metrics_header, append_metrics_row


def train(
    cfg: Config = None,
    mode: str = "linear_probe",
    backbone_checkpoint: str | Path = None,
):
    """
    Fine-tune a BYOL-pretrained ResNet34 backbone for cassava leaf classification.

    Args:
        cfg: Config object. Defaults to Config() if None.
        mode: 'linear_probe' (frozen backbone) or 'full_finetune' (differential LR).
        backbone_checkpoint: path to backbone_weights.pt. Defaults to
            cfg.checkpoint_path / 'backbone_weights.pt'.
    """
    if cfg is None:
        cfg = Config()

    if mode not in ("linear_probe", "full_finetune"):
        raise ValueError(f"mode must be 'linear_probe' or 'full_finetune', got '{mode}'")

    if cfg.use_weighted_ce and cfg.use_balanced_sampler:
        raise ConfigError(
            "use_weighted_ce and use_balanced_sampler cannot both be True."
        )

    set_seeds(cfg.seed)
    device = torch.device(cfg.device)

    checkpoint_path = cfg.checkpoint_path
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    if backbone_checkpoint is None:
        backbone_checkpoint = checkpoint_path / "backbone_weights.pt"

    train_df, val_df = make_splits(cfg)

    train_transform = build_train_transform(cfg)
    val_transform = build_val_transform(cfg)

    train_dataset = LeafDataset(train_df, cfg.train_images_path, transform=train_transform)
    val_dataset = LeafDataset(val_df, cfg.train_images_path, transform=val_transform)

    train_labels = train_df["label"].to_numpy()
    class_weights = compute_class_weights(train_labels, NUM_CLASSES).to(device)

    if cfg.use_balanced_sampler:
        sampler = make_balanced_sampler(train_labels, NUM_CLASSES)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    backbone = load_backbone_from_checkpoint(backbone_checkpoint, cfg.backbone)
    model = LeafClassifier(backbone, cfg.feature_dim, NUM_CLASSES).to(device)

    if mode == "linear_probe":
        model.freeze_backbone()
        optimizer = torch.optim.AdamW(
            model.head.parameters(),
            lr=cfg.finetune_lr_head,
            weight_decay=cfg.finetune_weight_decay,
        )
        print(
            f"Linear probe mode: backbone frozen, training head only "
            f"(lr={cfg.finetune_lr_head})."
        )
    else:
        model.unfreeze_backbone()
        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": cfg.finetune_lr_backbone},
                {"params": model.head.parameters(), "lr": cfg.finetune_lr_head},
            ],
            weight_decay=cfg.finetune_weight_decay,
        )
        print(
            f"Full fine-tune mode: backbone lr={cfg.finetune_lr_backbone}, "
            f"head lr={cfg.finetune_lr_head}."
        )

    if cfg.use_weighted_ce:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.finetune_epochs
    )

    metrics_path = Path(cfg.data_dir) / cfg.finetune_metrics_file
    write_metrics_header(metrics_path)

    best_val_accuracy = 0.0
    best_checkpoint = checkpoint_path / f"finetune_{mode}_best.pt"

    print(
        f"Fine-tuning ({mode}) on {len(train_dataset)} samples, "
        f"validating on {len(val_dataset)} samples."
    )

    for epoch in range(1, cfg.finetune_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(labels)

        train_loss = epoch_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        append_metrics_row(metrics_path, epoch, train_loss, val_metrics)

        print(
            f"Epoch {epoch:03d}/{cfg.finetune_epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  -> Saved best checkpoint (val_accuracy={best_val_accuracy:.4f})")

    print(f"Fine-tuning complete. Best val accuracy: {best_val_accuracy:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["linear_probe", "full_finetune"],
        default="linear_probe",
        help="Fine-tuning mode",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
        help="Path to backbone_weights.pt",
    )
    args = parser.parse_args()
    train(mode=args.mode, backbone_checkpoint=args.backbone_checkpoint)
