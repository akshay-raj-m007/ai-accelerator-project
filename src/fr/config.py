"""Configuration: dataclass + YAML + CLI overrides. No hard-coded machine paths.

The root directory comes from ``--root`` or the ``FR_ROOT`` environment variable
(falling back to the current directory). Layout under the root::

    data/                      CIFAR-10 (auto-downloaded)
    checkpoints/               e.g. resnet18_cifar10_checkpoint.pth (not in git)
    checkpoints/<run_name>/    last.pth, best.pth written by training
    results/<run_name>/        logs, CSV, reports
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

LEGACY_CKPT_NAME = "resnet18_cifar10_checkpoint.pth"


@dataclass
class Config:
    # paths
    root: Optional[str] = None
    data_dir: Optional[str] = None
    run_name: str = "run"
    init_checkpoint: Optional[str] = None
    # model / init / preprocessing
    arch: str = "resnet18_cifar"     # resnet18_cifar | mobilenetv2_cifar (see fr.model.ARCHS)
    init: str = "scratch"            # scratch | checkpoint | imagenet (mobilenetv2_cifar only: torchvision ImageNet weights)
    pretrained_weights: str = "IMAGENET1K_V1"   # init=imagenet: torchvision weight set; init_checkpoint = local state-dict file instead
    normalize: str = "auto"          # auto | none | cifar10
    # optimisation
    epochs: int = 200
    lr: float = 0.1
    batch_size: int = 128
    momentum: float = 0.9
    nesterov: bool = True
    weight_decay: float = 5e-4
    warmup_epochs: float = 5.0
    label_smoothing: float = 0.0
    amp: bool = True
    # data
    val_size: int = 5000
    train_subset: Optional[int] = None   # smoke tests only
    test_subset: Optional[int] = None
    num_workers: int = 4
    eval_batch_size: int = 500
    # misc
    seed: int = 0
    device: str = "auto"
    deterministic: bool = False
    resume: bool = False

    # ---- derived paths
    def root_path(self) -> Path:
        return Path(self.root or os.environ.get("FR_ROOT") or ".").expanduser().resolve()

    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser() if self.data_dir else self.root_path() / "data"

    def ckpt_dir(self) -> Path:
        return self.root_path() / "checkpoints" / self.run_name

    def results_dir(self) -> Path:
        return self.root_path() / "results" / self.run_name

    def init_ckpt_path(self) -> Path:
        if self.init_checkpoint:
            return Path(self.init_checkpoint).expanduser()
        return self.root_path() / "checkpoints" / LEGACY_CKPT_NAME

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_OPTIONAL_TYPES = {"root": str, "data_dir": str, "init_checkpoint": str,
                   "train_subset": int, "test_subset": int}
ARCH_CHOICES = ["resnet18_cifar", "mobilenetv2_cifar"]      # must equal fr.model.ARCHS (checked in the tests)
_CHOICES = {"arch": ARCH_CHOICES, "init": ["scratch", "checkpoint", "imagenet"], "normalize": ["auto", "none", "cifar10"],
            "pretrained_weights": ["IMAGENET1K_V1", "IMAGENET1K_V2"]}


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add one ``--flag`` per Config field (default None = "not given")."""
    parser.add_argument("--config", help="YAML file with Config fields")
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
            continue
        typ = _OPTIONAL_TYPES.get(f.name)
        if typ is None:
            typ = type(f.default)
        parser.add_argument(flag, type=typ, default=None, choices=_CHOICES.get(f.name))
    return parser


def config_from_args(args: argparse.Namespace, base: Optional[dict] = None) -> Config:
    """Precedence: CLI flag > YAML file > dataclass default."""
    values: dict = dict(base or {})
    if getattr(args, "config", None):
        with open(args.config, "r", encoding="utf-8") as fh:
            values.update(yaml.safe_load(fh) or {})
    names = {f.name for f in dataclasses.fields(Config)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    for name in names:
        v = getattr(args, name, None)
        if v is not None:
            values[name] = v
    return Config(**values)
