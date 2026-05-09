from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

from config import BACKBONE_FEATURE_DIM


def build_backbone(name: str, pretrained: bool = True) -> nn.Module:
    """
    Build a backbone with the classification head replaced by Identity.

    The resulting module outputs a feature vector of size BACKBONE_FEATURE_DIM[name].
    Raises KeyError for unsupported backbone names.
    """
    if name not in BACKBONE_FEATURE_DIM:
        raise KeyError(
            f"Unsupported backbone '{name}'. "
            f"Choose from: {list(BACKBONE_FEATURE_DIM.keys())}"
        )

    weights_arg = "DEFAULT" if pretrained else None

    if name == "resnet34":
        model = tvm.resnet34(weights=weights_arg)
        model.fc = nn.Identity()
    elif name == "resnet50":
        model = tvm.resnet50(weights=weights_arg)
        model.fc = nn.Identity()
    elif name == "efficientnet_b4":
        model = tvm.efficientnet_b4(weights=weights_arg)
        model.classifier = nn.Identity()
    elif name == "densenet121":
        model = tvm.densenet121(weights=weights_arg)
        model.classifier = nn.Identity()
    else:
        raise KeyError(f"Unsupported backbone '{name}'")

    return model


def load_backbone_from_checkpoint(checkpoint_path: str | Path, backbone_name: str) -> nn.Module:
    """
    Load backbone weights from a backbone_weights.pt checkpoint.

    The checkpoint must contain the backbone state_dict (not the full BYOL model).
    Raises RuntimeError on key mismatches rather than silently ignoring them.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {path}")

    backbone = build_backbone(backbone_name, pretrained=False)
    state_dict = torch.load(path, map_location="cpu")

    missing, unexpected = backbone.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint key mismatch loading '{path}'.\n"
            f"  Missing keys: {missing}\n"
            f"  Unexpected keys: {unexpected}"
        )

    return backbone


class ClassificationHead(nn.Module):
    """Linear classification head attached on top of a backbone."""

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class LeafClassifier(nn.Module):
    """Backbone + classification head for supervised training or fine-tuning."""

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True
