"""
Supervised baseline: ImageNet-pretrained ResNet34 fine-tuned for cassava leaf disease.
Establishes a performance benchmark before BYOL pretraining.
"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_fscore_support

from config import Config, NUM_CLASSES
from augmentations import build_train_transform, build_val_transform
from data_utils import (
    LeafDataset,
    compute_class_weights,
    make_balanced_sampler,
    make_splits,
    set_seeds,
)
from model_utils import LeafClassifier, build_backbone


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)), zero_division=0
    )

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_precision": per_class_p.tolist(),
        "per_class_recall": per_class_r.tolist(),
        "per_class_f1": per_class_f1.tolist(),
    }


def write_metrics_header(metrics_path: Path):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "train_loss", "val_loss", "val_accuracy", "macro_f1"]
        for i in range(NUM_CLASSES):
            header += [f"class{i}_precision", f"class{i}_recall", f"class{i}_f1"]
        writer.writerow(header)


def append_metrics_row(metrics_path: Path, epoch: int, train_loss: float, val_metrics: dict):
    with open(metrics_path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [
            epoch,
            f"{train_loss:.6f}",
            f"{val_metrics['loss']:.6f}",
            f"{val_metrics['accuracy']:.6f}",
            f"{val_metrics['macro_f1']:.6f}",
        ]
        for i in range(NUM_CLASSES):
            row += [
                f"{val_metrics['per_class_precision'][i]:.6f}",
                f"{val_metrics['per_class_recall'][i]:.6f}",
                f"{val_metrics['per_class_f1'][i]:.6f}",
            ]
        writer.writerow(row)


def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    set_seeds(cfg.seed)
    device = torch.device(cfg.device)

    checkpoint_path = cfg.checkpoint_path
    checkpoint_path.mkdir(parents=True, exist_ok=True)

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

    backbone = build_backbone(cfg.backbone, pretrained=True)
    model = LeafClassifier(backbone, cfg.feature_dim, NUM_CLASSES).to(device)

    if cfg.use_weighted_ce:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    metrics_path = Path(cfg.data_dir) / cfg.metrics_file
    write_metrics_header(metrics_path)

    best_val_accuracy = 0.0
    best_checkpoint = checkpoint_path / "baseline_best.pt"

    print(f"Training supervised baseline on {len(train_dataset)} samples, "
          f"validating on {len(val_dataset)} samples.")

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        append_metrics_row(metrics_path, epoch, train_loss, val_metrics)

        print(
            f"Epoch {epoch:03d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_checkpoint)
            print(f"  -> Saved best checkpoint (val_accuracy={best_val_accuracy:.4f})")

    print(f"Training complete. Best val accuracy: {best_val_accuracy:.4f}")
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Metrics CSV: {metrics_path}")


if __name__ == "__main__":
    train()
