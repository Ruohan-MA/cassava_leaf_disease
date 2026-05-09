"""
BYOL self-supervised pretraining on cassava leaf images.

Adapted from cancer_detection/ssl_experiment.py:
- Removed torchstain stain normalization (histopathology-specific)
- Updated CSV/path schema for cassava dataset
- Added projection std logging for collapse detection
- Uses train_images/ only (test images excluded via get_ssl_image_paths guard)
"""
import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.loss import NegativeCosineSimilarity

from config import Config
from augmentations import build_ssl_transform
from data_utils import LeafSSLDataset, get_ssl_image_paths, set_seeds
from model_utils import build_backbone


class BYOL(nn.Module):
    """
    BYOL model with online and target networks.

    The target network is updated via EMA; it never receives gradients.
    The backbone is accessible as self.backbone for checkpoint saving.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        feature_dim = cfg.feature_dim

        self.backbone = build_backbone(cfg.backbone, pretrained=True)

        self.projection_head = BYOLProjectionHead(
            input_dim=feature_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.projection_dim,
        )
        self.prediction_head = BYOLPredictionHead(
            input_dim=cfg.projection_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=cfg.projection_dim,
        )

        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_projection_head = copy.deepcopy(self.projection_head)

        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projection_head.parameters():
            param.requires_grad = False

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        z1_online = self.projection_head(self.backbone(x1))
        z2_online = self.projection_head(self.backbone(x2))
        p1 = self.prediction_head(z1_online)
        p2 = self.prediction_head(z2_online)

        with torch.no_grad():
            z1_target = self.target_projection_head(self.target_backbone(x1))
            z2_target = self.target_projection_head(self.target_backbone(x2))

        return p1, p2, z1_target.detach(), z2_target.detach(), z1_online, z2_online

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for online, target in zip(
            self.backbone.parameters(), self.target_backbone.parameters()
        ):
            target.data = momentum * target.data + (1.0 - momentum) * online.data

        for online, target in zip(
            self.projection_head.parameters(), self.target_projection_head.parameters()
        ):
            target.data = momentum * target.data + (1.0 - momentum) * online.data


def _write_ssl_metrics_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "proj_std", "collapse_warning"])


def _append_ssl_metrics_row(path: Path, epoch: int, loss: float, proj_std: float, threshold: float):
    collapsed = proj_std < threshold
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, f"{loss:.6f}", f"{proj_std:.6f}", int(collapsed)])
    return collapsed


def train(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    set_seeds(cfg.seed)
    device = torch.device(cfg.device)

    checkpoint_path = cfg.checkpoint_path
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    image_paths = get_ssl_image_paths(cfg)
    print(f"BYOL pretraining on {len(image_paths)} images from {cfg.train_images_path}")

    ssl_transform = build_ssl_transform(cfg)
    ssl_dataset = LeafSSLDataset(image_paths, transform=ssl_transform)
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = BYOL(cfg).to(device)
    criterion = NegativeCosineSimilarity()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.ssl_lr, weight_decay=cfg.ssl_weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.ssl_epochs
    )

    metrics_path = Path(cfg.data_dir) / cfg.ssl_metrics_file
    _write_ssl_metrics_header(metrics_path)

    print(f"Starting BYOL pretraining for {cfg.ssl_epochs} epochs.")

    for epoch in range(1, cfg.ssl_epochs + 1):
        model.train()
        epoch_loss = 0.0
        all_proj_vecs = []

        for view1, view2 in ssl_loader:
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()

            p1, p2, z1, z2, z1_online, z2_online = model(view1, view2)
            loss = 0.5 * (criterion(p1, z2) + criterion(p2, z1))
            loss.backward()
            optimizer.step()

            model.update_target(cfg.ema_momentum)
            epoch_loss += loss.item() * view1.size(0)

            with torch.no_grad():
                all_proj_vecs.append(z1_online.detach().cpu())

        scheduler.step()

        avg_loss = epoch_loss / len(ssl_loader.dataset)
        proj_tensor = torch.cat(all_proj_vecs, dim=0)
        proj_std = proj_tensor.std(dim=0).mean().item()

        collapsed = _append_ssl_metrics_row(
            metrics_path, epoch, avg_loss, proj_std, cfg.byol_collapse_std_threshold
        )

        status = " [COLLAPSE WARNING]" if collapsed else ""
        print(
            f"Epoch {epoch:03d}/{cfg.ssl_epochs}  "
            f"loss={avg_loss:.4f}  proj_std={proj_std:.4f}{status}"
        )

        if epoch % cfg.save_every == 0 or epoch == cfg.ssl_epochs:
            backbone_ckpt = checkpoint_path / f"backbone_weights_epoch{epoch:03d}.pt"
            torch.save(model.backbone.state_dict(), backbone_ckpt)
            # Overwrite the canonical latest checkpoint
            torch.save(
                model.backbone.state_dict(),
                checkpoint_path / "backbone_weights.pt",
            )
            print(f"  -> Saved backbone checkpoint at epoch {epoch}")

    print(f"BYOL pretraining complete.")
    print(f"Final backbone checkpoint: {checkpoint_path / 'backbone_weights.pt'}")
    print(f"SSL metrics: {metrics_path}")


if __name__ == "__main__":
    train()
