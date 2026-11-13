"""MobileNetV2 / CIFAR-10 port: builder, architecture selection, inventory, baseline checks, training wiring (CPU, no data download)."""
import numpy as np
import pytest
import torch
from torchvision import datasets

from fr import baseline_profile as BP
from fr import data as fdata
from fr import train as ftrain
from fr.config import ARCH_CHOICES, Config
from fr.model import (ARCHS, arch_of_checkpoint, build_mobilenetv2_cifar, build_model, build_resnet18_cifar, classifier_linear,
                      load_model, quantizable_modules, stem_conv)


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return build_mobilenetv2_cifar(10).eval()


def test_builder_changes_only_the_first_stride(net):
    from torchvision.models import mobilenet_v2
    ref = mobilenet_v2(weights=None, num_classes=10)
    assert tuple(ref.features[0][0].stride) == (2, 2)                     # torchvision's own stem stride
    assert tuple(stem_conv(net).stride) == (1, 1) and tuple(stem_conv(net).kernel_size) == (3, 3)
    assert classifier_linear(net).out_features == 10 and classifier_linear(net).in_features == 1280
    assert [k for k in net.state_dict()] == [k for k in ref.state_dict()]       # same parameter names and order
    assert all(a.shape == b.shape for a, b in zip(net.state_dict().values(), ref.state_dict().values()))
    strides = {n: tuple(m.stride) for n, m in net.named_modules() if isinstance(m, torch.nn.Conv2d) and m.stride != (1, 1)}
    assert len(strides) == 4 and all(v == (2, 2) for v in strides.values())    # stages 2, 3, 4, 6 keep their stride-2 depthwise conv
    with torch.no_grad():
        assert net(torch.randn(2, 3, 32, 32)).shape == (2, 10)
        assert build_mobilenetv2_cifar(100).eval()(torch.randn(1, 3, 32, 32)).shape == (1, 100)   # the CIFAR-100 owner's head


def test_final_feature_map_is_2x2(net):
    with torch.no_grad():
        assert net.features(torch.randn(1, 3, 32, 32)).shape == (1, 1280, 2, 2)


def test_layer_and_weight_counts(net):
    mods = quantizable_modules(net)
    assert len(mods) == 53 and sum(m.weight.numel() for _, m in mods) == 2_202_560
    rows = BP.layer_inventory(net)
    roles = {}
    for r in rows:
        roles[r["layer_role"]] = roles.get(r["layer_role"], 0) + 1
    assert roles == {"stem": 1, "depthwise": 17, "pointwise": 34, "classifier": 1}
    assert sum(r["num_weights"] for r in rows if r["layer_role"] == "depthwise") == 64_224
    assert sum(r["out_channels"] for r in rows) == 17_066                        # one float32 scale each if per-channel
    assert all(BP.reconcile(rows, net).values())
    ps = BP.parameter_summary(net, rows)
    assert ps["quantizable_plus_nonquantized_equals_total"] and ps["fp32_quantizable_bytes"] == 4 * 2_202_560


def test_groups_and_roles_by_name():
    assert BP.layer_group("features.0.0") == "stem" and BP.layer_group("features.1.conv.0.0") == "stage1"
    assert BP.layer_group("features.2.conv.0.0") == "stage2" and BP.layer_group("features.3.conv.2") == "stage2"
    assert BP.layer_group("features.17.conv.1.0") == "stage7" and BP.layer_group("features.18.0") == "head"
    assert BP.layer_group("classifier.1") == "classifier"
    # ResNet-18 behaviour is unchanged
    assert BP.layer_group("layer3.0.conv1") == "layer3" and BP.layer_group("fc") == "classifier" and BP.layer_group("conv1") == "stem"
    groups = {r["layer_group"] for r in BP.layer_inventory(build_mobilenetv2_cifar(10))}
    assert groups == {"stem", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "head", "classifier"}


def test_checkpoint_checks_work_for_both_architectures(net):
    c = BP.checkpoint_checks(net, {}, net.state_dict(), {})
    assert c["strict_load"] and c["output_dim_is_10"] and c["stem_is_3x3_stride1"] and c["no_maxpool"]
    assert c["n_quantizable_layers"] == 53 and c["n_quantizable_weights"] == 2_202_560
    r = BP.checkpoint_checks(build_resnet18_cifar().eval(), {}, build_resnet18_cifar().state_dict(), {})
    assert r["strict_load"] and r["stem_is_3x3_stride1"] and r["n_quantizable_layers"] == 21
    bad = dict(net.state_dict())
    bad.pop("classifier.1.bias")
    assert not BP.checkpoint_checks(net, {}, bad, {})["strict_load"]


def test_config_choices_match_registry():
    assert sorted(ARCH_CHOICES) == sorted(ARCHS) and Config().arch == "resnet18_cifar"
    with pytest.raises(ValueError):
        build_model("vgg16")


def test_load_model_picks_the_architecture_from_the_checkpoint(net, tmp_path):
    ck = {"epoch": 3, "model_state_dict": net.state_dict(), "config": {"arch": "mobilenetv2_cifar", "seed": 0, "val_size": 5000}}
    torch.save(ck, tmp_path / "m.pth")
    model, back = load_model(tmp_path / "m.pth")
    assert arch_of_checkpoint(back) == "mobilenetv2_cifar" and len(quantizable_modules(model)) == 53
    legacy = {"epoch": 1, "model_state_dict": build_resnet18_cifar().state_dict()}          # no config at all -> ResNet-18
    torch.save(legacy, tmp_path / "r.pth")
    assert len(quantizable_modules(load_model(tmp_path / "r.pth")[0])) == 21
    with pytest.raises(RuntimeError):                                                      # strict load: wrong architecture fails
        load_model(tmp_path / "r.pth", arch="mobilenetv2_cifar")


@pytest.fixture
def fake_cifar(monkeypatch):
    def fake(root, train, transform, download=True):
        return datasets.FakeData(size=96 if train else 48, image_size=(3, 32, 32), num_classes=10,
                                 transform=transform, random_offset=0 if train else 10_000)
    monkeypatch.setattr(fdata, "_cifar10", fake)


def test_training_records_and_reloads_the_architecture(tmp_path, fake_cifar):
    cfg = Config(root=str(tmp_path), run_name="mnv2", arch="mobilenetv2_cifar", init="scratch", epochs=1, lr=0.05, batch_size=32,
                 val_size=32, train_subset=64, test_subset=40, num_workers=0, amp=False, device="cpu", warmup_epochs=0.5,
                 eval_batch_size=32)
    out = ftrain.train(cfg)
    assert out["final_checkpoint"]["test"]["total"] == 40
    ck = torch.load(tmp_path / "checkpoints/mnv2/best.pth", weights_only=False)
    assert ck["config"]["arch"] == "mobilenetv2_cifar"
    model, _ = load_model(tmp_path / "checkpoints/mnv2/best.pth")
    assert len(quantizable_modules(model)) == 53
    assert any(np.isfinite(p.detach().numpy()).all() for p in model.parameters())


# ------------------------------------------------------------------------------------ ImageNet-weight fine-tuning start
@pytest.fixture(scope="module")
def imagenet_like_file(tmp_path_factory):
    """A stand-in for torchvision's ImageNet checkpoint (same keys/shapes, 1000 classes, random values): no download in tests."""
    from torchvision.models import mobilenet_v2
    torch.manual_seed(1)
    sd = mobilenet_v2(weights=None).state_dict()
    path = tmp_path_factory.mktemp("w") / "mobilenet_v2_imagenet_like.pth"
    torch.save(sd, path)
    return path, sd


def test_imagenet_weights_fill_everything_but_the_classifier(imagenet_like_file):
    from fr.model import load_imagenet_weights
    path, sd = imagenet_like_file
    m = build_mobilenetv2_cifar(10)
    head_before = {k: v.clone() for k, v in m.state_dict().items() if k.startswith("classifier.")}
    info = load_imagenet_weights(m, path=path)
    now = m.state_dict()
    assert info["num_tensors_loaded"] == len(sd) - 2 and info["classifier_reinitialised"] == ["classifier.1.bias", "classifier.1.weight"]
    assert len(info["file_sha256"]) == 64
    for k, v in sd.items():
        if k.startswith("classifier."):
            assert torch.equal(now[k], head_before[k]) and now[k].shape[0] == 10        # fresh 10-class head, untouched
        else:
            assert torch.equal(now[k], v), k                                            # stem (stride 1 now) and all blocks copied
    assert tuple(stem_conv(m).stride) == (1, 1)


def test_imagenet_weights_reject_a_wrong_state_dict(tmp_path):
    from fr.model import load_imagenet_weights
    torch.save(build_resnet18_cifar().state_dict(), tmp_path / "r.pth")
    with pytest.raises(RuntimeError):
        load_imagenet_weights(build_mobilenetv2_cifar(10), path=tmp_path / "r.pth")


def test_training_from_imagenet_weights_records_the_source(tmp_path, fake_cifar, imagenet_like_file):
    path, _ = imagenet_like_file
    cfg = Config(root=str(tmp_path), run_name="ft", arch="mobilenetv2_cifar", init="imagenet", init_checkpoint=str(path), epochs=1,
                 lr=0.02, batch_size=32, val_size=32, train_subset=64, test_subset=40, num_workers=0, amp=False, device="cpu",
                 warmup_epochs=0.5, eval_batch_size=32)
    ftrain.train(cfg)
    ck = torch.load(tmp_path / "checkpoints/ft/best.pth", weights_only=False)
    ps = ck["config"]["pretrained_source"]
    assert ck["config"]["init"] == "imagenet" and ps["num_tensors_loaded"] == 312 and ps["file"] == str(path)
    assert ck["config"]["normalization"] == fdata.normalization_to_config((fdata.CIFAR10_MEAN, fdata.CIFAR10_STD))
    assert ck["config"]["arch"] == "mobilenetv2_cifar"
    with pytest.raises(ValueError):                                                     # only implemented for MobileNetV2
        ftrain.train(Config(root=str(tmp_path), run_name="bad", arch="resnet18_cifar", init="imagenet", device="cpu"))


def test_cli_and_yaml_accept_the_imagenet_init():
    import argparse
    from fr.config import add_config_args, config_from_args
    ap = add_config_args(argparse.ArgumentParser())
    cfg = config_from_args(ap.parse_args(["--config", "configs/finetune_mobilenetv2_imagenet.yaml"]))
    assert cfg.arch == "mobilenetv2_cifar" and cfg.init == "imagenet" and cfg.pretrained_weights == "IMAGENET1K_V1"
    assert config_from_args(ap.parse_args(["--init", "imagenet", "--pretrained-weights", "IMAGENET1K_V2"])).pretrained_weights == "IMAGENET1K_V2"
