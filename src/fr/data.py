"""CIFAR-10 loaders and the normalization policy.

Normalization is *explicit state*: it is stored in every checkpoint that
``train.py`` writes (``config["normalization"]``) and must be determined from
``checkpoint["config"]`` for externally produced checkpoints. If it cannot be
determined we raise :class:`NormalizationUnknown` rather than guess.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

Norm = Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]   # None => ToTensor() only


class NormalizationUnknown(RuntimeError):
    pass


def _cifar10(root, train: bool, transform, download: bool = True):
    """Single construction point (tests monkeypatch this)."""
    return datasets.CIFAR10(root=str(root), train=train, transform=transform, download=download)


def make_transform(norm: Norm, train: bool):
    ops = []
    if train:  # paper-agnostic standard CIFAR augmentation requested by the task
        ops += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    ops.append(transforms.ToTensor())
    if norm is not None:
        ops.append(transforms.Normalize(norm[0], norm[1]))
    return transforms.Compose(ops)


def _find(d, keys):
    """Depth-first search of a nested dict for the first key (case-insens.) in keys."""
    if isinstance(d, dict):
        for k, v in d.items():
            if str(k).lower() in keys:
                return v
        for v in d.values():
            r = _find(v, keys)
            if r is not None:
                return r
    return None


def norm_from_checkpoint_config(config) -> Norm | str:
    """Return a Norm, or the string ``"unknown"`` if the config does not say.

    Recognised: our own ``normalization`` entry ({"mean","std"} or None);
    otherwise explicit ``mean``/``std`` lists, or a boolean ``normalize``
    flag set to False. A bare ``normalize: True`` without statistics is *not*
    accepted (which constants?).
    """
    if not isinstance(config, dict):
        return "unknown"
    if "normalization" in config:
        n = config["normalization"]
        return None if n is None else (tuple(n["mean"]), tuple(n["std"]))
    mean = _find(config, {"mean", "norm_mean", "normalize_mean", "cifar_mean"})
    std = _find(config, {"std", "norm_std", "normalize_std", "cifar_std"})
    if mean is not None and std is not None:
        return (tuple(float(x) for x in mean), tuple(float(x) for x in std))
    tt = config.get("training_transform")
    if isinstance(tt, str) and tt.strip().lower() == "totensor":
        return None   # legacy checkpoint: config says it was trained with ToTensor() only
    flag = _find(config, {"normalize", "normalization", "use_normalization"})
    if flag is False:
        return None
    return "unknown"


def resolve_normalization(policy: str, init: str, ckpt_config=None) -> Norm:
    """policy: auto | none | cifar10 (explicit user choices always win)."""
    if policy == "none":
        return None
    if policy == "cifar10":
        return (CIFAR10_MEAN, CIFAR10_STD)
    if init == "scratch" and ckpt_config is None:
        return (CIFAR10_MEAN, CIFAR10_STD)
    found = norm_from_checkpoint_config(ckpt_config)
    if found == "unknown":
        raise NormalizationUnknown(
            "Cannot determine the input normalization from checkpoint['config'].\n"
            "Run `python scripts/inspect_checkpoint.py <ckpt>` and tell the maintainer, or pass "
            "--normalize none (ToTensor only, as the old notebook evaluated) or --normalize cifar10 "
            "explicitly. Use `scripts/eval_fp32.py --diagnose-normalization` to compare both on the "
            "test set (diagnostic only; never use the test set to pick preprocessing for a paper number)."
        )
    return found


def normalization_to_config(norm: Norm):
    return None if norm is None else {"mean": list(norm[0]), "std": list(norm[1])}


def get_test_loader(data_dir, norm: Norm, batch_size: int, num_workers: int = 0,
                    subset: Optional[int] = None, pin_memory: bool = False) -> DataLoader:
    ds = _cifar10(data_dir, False, make_transform(norm, train=False))
    if subset:
        ds = Subset(ds, range(min(subset, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory)


def split_indices(n: int, val_size: int, seed: int):
    """Fixed train/val split, independent of the training RNG."""
    g = torch.Generator().manual_seed(1234 + seed)
    perm = torch.randperm(n, generator=g).tolist()
    return perm[val_size:], perm[:val_size]


def get_val_loader(data_dir, norm: Norm, eval_batch_size: int, val_size: int, seed: int,
                   num_workers: int = 0, pin_memory: bool = False, subset: Optional[int] = None,
                   skip: int = 0):
    """The held-out validation split only (no augmentation), same indices as ``get_train_val_loaders``.

    ``skip`` drops the first ``skip`` validation images (they were used for something else, e.g. a held-out
    Fisher estimate); ``subset`` then keeps the first ``subset`` of the remaining ones."""
    plain = _cifar10(data_dir, True, make_transform(norm, train=False))
    _, va_idx = split_indices(len(plain), val_size, seed)
    if skip:
        va_idx = va_idx[skip:]
    if subset:
        va_idx = va_idx[:subset]
    return DataLoader(Subset(plain, va_idx), batch_size=eval_batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=pin_memory)


def get_calibration_loader(data_dir, norm: Norm, n: int, val_size: int, seed: int,
                           batch_size: int = 1, num_workers: int = 0):
    """``n`` un-augmented images from the *training* part of the split (never validation or test).

    Used for Fisher information, so no test or validation image influences the sensitivity scores.
    ``val_size``/``seed`` must equal the values used when the checkpoint was trained.
    """
    plain = _cifar10(data_dir, True, make_transform(norm, train=False))
    tr_idx, _ = split_indices(len(plain), val_size, seed)
    return DataLoader(Subset(plain, tr_idx[:n]), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers)


def get_train_val_loaders(data_dir, norm: Norm, batch_size: int, eval_batch_size: int,
                          val_size: int, seed: int, num_workers: int = 0,
                          train_subset: Optional[int] = None, pin_memory: bool = False):
    """Train (augmented, shuffled) and held-out validation loaders from the 50k train set."""
    train_aug = _cifar10(data_dir, True, make_transform(norm, train=True))
    train_plain = _cifar10(data_dir, True, make_transform(norm, train=False))
    tr_idx, va_idx = split_indices(len(train_aug), val_size, seed)
    if train_subset:
        tr_idx = tr_idx[:train_subset]
    train_loader = DataLoader(Subset(train_aug, tr_idx), batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader = DataLoader(Subset(train_plain, va_idx), batch_size=eval_batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader
