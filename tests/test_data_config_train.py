import argparse

import pytest
import torch
from torchvision import datasets

from fr import data as fdata
from fr import train as ftrain
from fr.config import Config, add_config_args, config_from_args
from fr.data import NormalizationUnknown, resolve_normalization


def test_normalization_policy():
    assert resolve_normalization("none", "checkpoint", {}) is None
    assert resolve_normalization("cifar10", "checkpoint", {})[0] == fdata.CIFAR10_MEAN
    assert resolve_normalization("auto", "scratch", None)[0] == fdata.CIFAR10_MEAN
    own = {"normalization": {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}}
    assert resolve_normalization("auto", "checkpoint", own) == ((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    assert resolve_normalization("auto", "checkpoint", {"normalization": None}) is None
    assert resolve_normalization("auto", "checkpoint", {"data": {"normalize": False}}) is None
    nested = {"transforms": {"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]}}
    assert resolve_normalization("auto", "checkpoint", nested)[1] == (0.2, 0.2, 0.2)
    for unknown in ({}, {"lr": 0.1, "epochs": 5}, {"normalize": True}, None):
        with pytest.raises(NormalizationUnknown):
            resolve_normalization("auto", "checkpoint", unknown)


def test_config_precedence(tmp_path):
    y = tmp_path / "c.yaml"
    y.write_text("epochs: 7\nlr: 0.5\ninit: checkpoint\n")
    ap = add_config_args(argparse.ArgumentParser())
    cfg = config_from_args(ap.parse_args(["--config", str(y), "--lr", "0.25", "--no-amp"]))
    assert (cfg.epochs, cfg.lr, cfg.init, cfg.amp) == (7, 0.25, "checkpoint", False)
    y.write_text("bogus: 1\n")
    with pytest.raises(ValueError):
        config_from_args(ap.parse_args(["--config", str(y)]))


def test_paths_from_root(tmp_path, monkeypatch):
    monkeypatch.setenv("FR_ROOT", str(tmp_path))
    cfg = Config(run_name="r")
    assert cfg.root_path() == tmp_path.resolve()
    assert cfg.ckpt_dir() == tmp_path.resolve() / "checkpoints" / "r"
    assert cfg.data_path() == tmp_path.resolve() / "data"
    assert Config(root=str(tmp_path / "x")).root_path() == (tmp_path / "x").resolve()


def test_split_is_fixed_and_disjoint():
    a = fdata.split_indices(50_000, 5000, 0)
    b = fdata.split_indices(50_000, 5000, 0)
    assert a == b and len(a[1]) == 5000 and not set(a[0]) & set(a[1])


def test_train_transform_has_crop_and_flip():
    names = [type(t).__name__ for t in fdata.make_transform(None, True).transforms]
    assert names[:2] == ["RandomCrop", "RandomHorizontalFlip"]
    names = [type(t).__name__ for t in fdata.make_transform(None, False).transforms]
    assert names == ["ToTensor"]


@pytest.fixture
def fake_cifar(monkeypatch):
    def fake(root, train, transform, download=True):
        return datasets.FakeData(size=160 if train else 80, image_size=(3, 32, 32), num_classes=10,
                                 transform=transform, random_offset=0 if train else 10_000)
    monkeypatch.setattr(fdata, "_cifar10", fake)


def _cfg(tmp_path, name, **kw):
    base = dict(root=str(tmp_path), run_name=name, init="scratch", epochs=2, lr=0.05, batch_size=32,
                val_size=32, train_subset=64, test_subset=40, num_workers=0, amp=False,
                device="cpu", warmup_epochs=1, eval_batch_size=32)
    base.update(kw)
    return Config(**base)


def test_train_and_exact_resume(tmp_path, monkeypatch, fake_cifar):
    full = ftrain.train(_cfg(tmp_path, "full"))
    assert (tmp_path / "checkpoints/full/last.pth").exists() and (tmp_path / "checkpoints/full/best.pth").exists()
    assert (tmp_path / "results/full/train_log.csv").read_text().splitlines()[0] == \
        "epoch,lr,train_loss,train_acc,val_loss,val_acc,time_s"
    assert full["final_checkpoint"]["test"]["total"] == 40

    # interrupt run B right after epoch 1 has been saved, then resume
    real_save, calls = ftrain._atomic_save, {"n": 0}

    def flaky(obj, path):
        calls["n"] += 1
        if calls["n"] > 2:                       # epoch 1 wrote last + best (first epoch is best)
            raise KeyboardInterrupt
        real_save(obj, path)

    monkeypatch.setattr(ftrain, "_atomic_save", flaky)
    with pytest.raises(KeyboardInterrupt):
        ftrain.train(_cfg(tmp_path, "part"))
    monkeypatch.setattr(ftrain, "_atomic_save", real_save)
    assert torch.load(tmp_path / "checkpoints/part/last.pth", weights_only=False)["epoch"] == 1
    ftrain.train(_cfg(tmp_path, "part", resume=True, epochs=999, lr=9.9))   # saved hyper-params win

    a = torch.load(tmp_path / "checkpoints/full/last.pth", weights_only=False)
    b = torch.load(tmp_path / "checkpoints/part/last.pth", weights_only=False)
    assert a["epoch"] == b["epoch"] == 2
    for k in a["model_state_dict"]:
        assert torch.equal(a["model_state_dict"][k], b["model_state_dict"][k]), k
    sa, sb = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
    for k in sa:
        assert torch.equal(sa[k]["momentum_buffer"], sb[k]["momentum_buffer"])
    assert a["config"]["normalization"] == b["config"]["normalization"]
    rows = (tmp_path / "results/part/train_log.csv").read_text().splitlines()
    assert len(rows) == 3                         # header + 2 epochs, no duplicates


def test_finetune_requires_known_normalization(tmp_path, fake_cifar):
    from fr.model import build_resnet18_cifar
    (tmp_path / "checkpoints").mkdir()
    ck = {"epoch": 1, "model_state_dict": build_resnet18_cifar().state_dict(),
          "optimizer_state_dict": {}, "config": {"lr": 0.1}}
    torch.save(ck, tmp_path / "checkpoints" / "resnet18_cifar10_checkpoint.pth")
    with pytest.raises(NormalizationUnknown):
        ftrain.train(_cfg(tmp_path, "ft", init="checkpoint"))
    out = ftrain.train(_cfg(tmp_path, "ft", init="checkpoint", normalize="none", epochs=1))
    assert out["normalization"] is None
