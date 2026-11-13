"""CIFAR variants of the models used in the J-FAMP study.

``resnet18_cifar``    torchvision ResNet-18 with a 3x3/s1 stem and no max-pool (identical to the original notebook).
``mobilenetv2_cifar`` torchvision MobileNetV2 (width 1.0) whose FIRST feature convolution has stride 1 instead of 2 and
                      whose classifier has ``num_classes`` outputs; nothing else differs from torchvision (protocol
                      "JFAMP_MobileNetV2_CIFAR10_CIFAR100_Experimental_Protocol", Section 1). On 32x32 inputs the
                      remaining strides give a 2x2 final feature map.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, resnet18

DEFAULT_ARCH = "resnet18_cifar"


def build_resnet18_cifar(num_classes: int = 10) -> nn.Module:
    """torchvision ResNet-18 with conv1 3x3/s1/p1 (no bias), no maxpool, fc=num_classes."""
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_mobilenetv2_cifar(num_classes: int = 10) -> nn.Module:
    """torchvision MobileNetV2 for 32x32 inputs: only the stride of the first feature convolution changes (2 -> 1).

    The stride is changed on the existing module (not by creating a new one) so that the weight initialisation of
    that layer is exactly torchvision's. The classifier is torchvision's ``Sequential(Dropout(0.2), Linear(1280, K))``.
    """
    model = mobilenet_v2(weights=None, num_classes=num_classes)
    stem = model.features[0][0]
    assert isinstance(stem, nn.Conv2d) and tuple(stem.stride) == (2, 2), "unexpected torchvision MobileNetV2 stem"
    stem.stride = (1, 1)
    return model


ARCHS = {"resnet18_cifar": build_resnet18_cifar, "mobilenetv2_cifar": build_mobilenetv2_cifar}


def build_model(arch: str = DEFAULT_ARCH, num_classes: int = 10) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture {arch!r}; choose from {sorted(ARCHS)}")
    return ARCHS[arch](num_classes)


def arch_of_checkpoint(ckpt: dict) -> str:
    """Architecture recorded by ``train.py`` in ``checkpoint['config']['arch']``; checkpoints without the entry
    (the legacy ResNet-18 notebook checkpoint) are ResNet-18."""
    arch = (ckpt.get("config") or {}).get("arch") if isinstance(ckpt, dict) else None
    return arch if arch in ARCHS else DEFAULT_ARCH


def stem_conv(model: nn.Module) -> nn.Conv2d:
    """The first convolution (``conv1`` of ResNet, ``features[0][0]`` of MobileNetV2)."""
    return model.conv1 if hasattr(model, "conv1") else model.features[0][0]


def classifier_linear(model: nn.Module) -> nn.Linear:
    """The final Linear layer (``fc`` of ResNet, ``classifier[-1]`` of MobileNetV2)."""
    return model.fc if hasattr(model, "fc") else model.classifier[-1]


def load_imagenet_weights(model: nn.Module, weights: str = "IMAGENET1K_V1", path=None) -> dict:
    """Initialise a ``mobilenetv2_cifar`` model from torchvision's ImageNet MobileNetV2 weights (fine-tuning start point).

    Every tensor except the 1000-class classifier is copied: the stem's stride is not a parameter, so all ``features.*``
    tensors have the shapes of the stride-2 ImageNet model. The classifier keeps torchvision's fresh initialisation
    (normal(0, 0.01), zero bias). ``path`` is a local file holding a torchvision state dict (offline machines); otherwise
    the file is downloaded by torchvision (hash-checked) into the torch hub cache. Returns a record for the checkpoint config."""
    from torchvision.models import MobileNet_V2_Weights
    if path:
        sd = torch.load(Path(path).expanduser(), map_location="cpu", weights_only=True)
        import hashlib
        h = hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()
        source = {"file": str(Path(path).expanduser()), "file_sha256": h}
    else:
        wenum = MobileNet_V2_Weights[weights]
        sd = wenum.get_state_dict(progress=True, check_hash=True)
        source = {"weights": weights, "url": wenum.url, "imagenet_top1_percent": wenum.meta["_metrics"]["ImageNet-1K"]["acc@1"]}
    body = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(body, strict=False)
    head = sorted(k for k in model.state_dict() if k.startswith("classifier."))
    if unexpected or sorted(missing) != head:
        raise RuntimeError(f"ImageNet weights do not fit the model: unexpected={list(unexpected)} missing={list(missing)}")
    return {"init": "imagenet", **source, "num_tensors_loaded": len(body), "classifier_reinitialised": head,
            "note": "all layers are fine-tuned on CIFAR-10; the stem runs with stride 1 (ImageNet model: stride 2)"}


def load_checkpoint(path, map_location="cpu") -> dict:
    # Own/trusted files only: checkpoints hold optimizer state and RNG arrays.
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def state_dict_of(ckpt: dict) -> dict:
    for key in ("model_state_dict", "model"):
        if key in ckpt:
            return ckpt[key]
    raise KeyError(f"No model weights in checkpoint; keys: {list(ckpt)}")


def load_model(path, num_classes: int = 10, map_location="cpu", arch: str | None = None):
    """Build the model and load the checkpoint with strict=True. Returns (model, ckpt).

    ``arch`` defaults to the architecture recorded in the checkpoint (ResNet-18 when none is recorded)."""
    ckpt = load_checkpoint(path, map_location)
    model = build_model(arch or arch_of_checkpoint(ckpt), num_classes)
    model.load_state_dict(state_dict_of(ckpt), strict=True)
    return model, ckpt


def quantizable_modules(model: nn.Module):
    """(name, module) for every Conv2d/Linear, in definition order."""
    return [(n, m) for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
