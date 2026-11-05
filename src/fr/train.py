"""Train from scratch or fine-tune the existing checkpoint.

* SGD (momentum 0.9, Nesterov), weight decay 5e-4, linear warm-up + cosine LR
  (stepped per iteration), AMP on CUDA, optional label smoothing.
* 5,000 training images are held out for validation / model selection. The test
  set is touched only once, at the very end, to *report* the final and the
  best-on-validation checkpoints; it is never used for selection.
* ``last.pth`` and ``best.pth`` are written every epoch (model, optimizer,
  scheduler, scaler, epoch, RNG states, full config, git commit). ``--resume``
  restores all of it and continues the same trajectory.
"""
from __future__ import annotations

import csv
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import data as fdata
from .config import Config
from .evaluate import evaluate
from .model import build_resnet18_cifar, load_checkpoint, state_dict_of
from .utils import dump_json, env_info, git_info, resolve_device, seed_everything

log = logging.getLogger("fr.train")
CSV_FIELDS = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "time_s"]


def make_lr_lambda(warmup_iters: int, total_iters: int):
    def fn(step: int) -> float:
        if warmup_iters > 0 and step < warmup_iters:
            return (step + 1) / warmup_iters
        p = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))
    return fn


def _rng_state() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"].cpu())
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in s["cuda"]])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _setup_logging(results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    root = logging.getLogger("fr")
    root.setLevel(logging.INFO)
    root.handlers.clear()
    for h in (logging.StreamHandler(), logging.FileHandler(results_dir / "train.log")):
        h.setFormatter(fmt)
        root.addHandler(h)


def _read_csv_upto(path: Path, epoch: int):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if int(r["epoch"]) <= epoch]


def _write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def _config_from_saved(saved: dict, current: Config) -> Config:
    names = set(Config.__dataclass_fields__)
    kept = {k: v for k, v in saved.items() if k in names}
    # machine-specific fields come from the current invocation
    for k in ("root", "data_dir", "device", "num_workers", "run_name", "init_checkpoint"):
        kept[k] = getattr(current, k)
    kept["resume"] = True
    return Config(**kept)


def train(cfg: Config) -> dict:
    device = resolve_device(cfg.device)
    ckpt_dir, res_dir = cfg.ckpt_dir(), cfg.results_dir()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(res_dir)
    last_path, best_path = ckpt_dir / "last.pth", ckpt_dir / "best.pth"

    resume_ckpt = None
    if cfg.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        resume_ckpt = load_checkpoint(last_path)
        cfg = _config_from_saved(resume_ckpt["config"], cfg)
        norm = fdata.norm_from_checkpoint_config(resume_ckpt["config"])
        log.info("Resuming from %s (epoch %d); hyper-parameters taken from the checkpoint",
                 last_path, resume_ckpt["epoch"])
    else:
        init_ckpt = None
        if cfg.init == "checkpoint":
            path = cfg.init_ckpt_path()
            if not path.exists():
                raise FileNotFoundError(f"Init checkpoint not found: {path} (see README, Part D)")
            init_ckpt = load_checkpoint(path)
        norm = fdata.resolve_normalization(
            cfg.normalize, cfg.init, None if init_ckpt is None else init_ckpt.get("config"))

    seed_everything(cfg.seed, cfg.deterministic)
    env = env_info()
    log.info("env: %s", env)
    pin = device.type == "cuda"
    train_loader, val_loader = fdata.get_train_val_loaders(
        cfg.data_path(), norm, cfg.batch_size, cfg.eval_batch_size, cfg.val_size, cfg.seed,
        cfg.num_workers, cfg.train_subset, pin)
    iters_per_epoch = len(train_loader)
    if iters_per_epoch == 0:
        raise ValueError("Training loader is empty (train_subset < batch_size?)")

    model = build_resnet18_cifar(10)
    if resume_ckpt is None and cfg.init == "checkpoint":
        model.load_state_dict(state_dict_of(init_ckpt), strict=True)
        log.info("Initialised from %s (fine-tuning)", cfg.init_ckpt_path())
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                                nesterov=cfg.nesterov, weight_decay=cfg.weight_decay)
    total_iters = cfg.epochs * iters_per_epoch
    warm_iters = min(int(cfg.warmup_epochs * iters_per_epoch), max(total_iters - 1, 0))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(warm_iters, total_iters))
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    config_blob = {**cfg.to_dict(), "normalization": fdata.normalization_to_config(norm),
                   "arch": "resnet18_cifar", "warmup_iters": warm_iters,
                   "iters_per_epoch": iters_per_epoch}
    start_epoch, best_val, best_loss = 0, -1.0, math.inf
    rows = []
    csv_path = res_dir / "train_log.csv"
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        scaler.load_state_dict(resume_ckpt["scaler_state_dict"])
        start_epoch = resume_ckpt["epoch"]
        best_val, best_loss = resume_ckpt["best_val_acc"], resume_ckpt["best_val_loss"]
        _set_rng_state(resume_ckpt["rng_state"])
        rows = _read_csv_upto(csv_path, start_epoch)
        config_blob = resume_ckpt["config"]
    dump_json({"config": config_blob, "env": env}, res_dir / "run_meta.json")
    log.info("config: %s", config_blob)

    for epoch in range(start_epoch + 1, cfg.epochs + 1):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "lr": lr_now, "train_loss": loss_sum / seen,
               "train_acc": 100.0 * correct / seen, "val_loss": val["cross_entropy"],
               "val_acc": val["accuracy_percent"], "time_s": time.time() - t0}
        rows.append(row)
        _write_csv(csv_path, rows)
        log.info("epoch %d/%d lr=%.5f train %.4f/%.2f%% val %.4f/%.2f%% (%.1fs)", epoch,
                 cfg.epochs, lr_now, row["train_loss"], row["train_acc"], row["val_loss"],
                 row["val_acc"], row["time_s"])

        is_best = val["accuracy_percent"] > best_val or (
            val["accuracy_percent"] == best_val and val["cross_entropy"] < best_loss)
        if is_best:
            best_val, best_loss = val["accuracy_percent"], val["cross_entropy"]
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": _rng_state(),
            "config": config_blob,
            "git_commit": git_info(),
            "train_accuracy": row["train_acc"],
            "average_loss": row["train_loss"],
            "val_acc": row["val_acc"],
            "best_val_acc": best_val,
            "best_val_loss": best_loss,
        }
        _atomic_save(state, last_path)
        if is_best:
            _atomic_save(state, best_path)

    # ---- final reporting on the test set (never used for selection)
    test_loader = fdata.get_test_loader(cfg.data_path(), norm, cfg.eval_batch_size,
                                        cfg.num_workers, cfg.test_subset, pin)
    final = evaluate(model, test_loader, device)
    best_model = build_resnet18_cifar(10)
    best_ckpt = load_checkpoint(best_path)
    best_model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    best = evaluate(best_model, test_loader, device)
    summary = {
        "run_name": cfg.run_name, "init": cfg.init, "epochs": cfg.epochs,
        "test_subset": cfg.test_subset,
        "final_checkpoint": {"path": str(last_path), "epoch": cfg.epochs, "test": final},
        "best_on_val_checkpoint": {"path": str(best_path), "epoch": best_ckpt["epoch"],
                                   "val_acc": best_ckpt["val_acc"], "test": best},
        "normalization": fdata.normalization_to_config(norm), "env": env,
    }
    dump_json(summary, res_dir / "final_metrics.json")
    log.info("TEST final: %.2f%%  | TEST best-on-val (epoch %d): %.2f%%",
             final["accuracy_percent"], best_ckpt["epoch"], best["accuracy_percent"])
    return summary
