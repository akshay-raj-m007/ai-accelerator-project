"""Accuracy / correct / total / mean cross-entropy (same fields as the notebook)."""
from __future__ import annotations

import torch
import torch.nn as nn


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device) -> dict:
    was_training = model.training
    model.eval()
    model.to(device)
    correct = total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits.float(), labels, reduction="sum").item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    model.train(was_training)
    return {
        "accuracy_percent": 100.0 * correct / total,
        "correct": correct,
        "total": total,
        "cross_entropy": loss_sum / total,
    }
