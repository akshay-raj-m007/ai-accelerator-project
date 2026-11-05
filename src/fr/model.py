"""ResNet-18, CIFAR variant (identical to the original notebook)."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-18 with conv1 3x3/s1/p1 (no bias), no maxpool, fc=num_classes."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path, map_location="cpu") -> dict:
    # Own/trusted files only: checkpoints hold optimizer state and RNG arrays.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def state_dict_of(ckpt: dict) -> dict:
    for key in ("model_state_dict", "model"):
        if key in ckpt:
            return ckpt[key]
    raise KeyError(f"No model weights in checkpoint; keys: {list(ckpt)}")


def load_model(path, num_classes: int = 10, map_location="cpu"):
    """Build the model and load the checkpoint with strict=True. Returns (model, ckpt)."""
    ckpt = load_checkpoint(path, map_location)
    model = build_resnet18_cifar(num_classes)
    model.load_state_dict(state_dict_of(ckpt), strict=True)
    return model, ckpt


def quantizable_modules(model: nn.Module):
    """(name, module) for every Conv2d/Linear, in definition order."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
