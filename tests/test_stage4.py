import json
import math
from collections import OrderedDict

import numpy as np
import pytest
import torch

from fr import stage2 as S2
from fr import stage3 as S3
from fr import stage4 as S4
from fr.model import build_resnet18_cifar, quantizable_modules

NAMES = [n for n, _ in quantizable_modules(build_resnet18_cifar())]
CODECS = S2.make_codecs(True)
CTX = {k: "x" for k in ("experiment_id", "run_id", "model", "dataset", "checkpoint_hash", "codec_version")}


def test_configuration_counts_and_skips():
    cfg, sk = S4.enumerate_configs("fp16", NAMES)
    assert len(cfg) == 4 * (2 + 21 * 2) == 176 and not any(c["scope_kind"] == "scales" for c in cfg)
    assert {c["group"] for c in cfg} == {"sign", "exponent", "fraction", "all"}
    cfg, sk = S4.enumerate_configs("int8_pt", NAMES)
    w = [c for c in cfg if c["scope_kind"] == "weights"]
    s = [c for c in cfg if c["scope_kind"] == "scales"]
    assert len(w) == 3 * (2 + 21 * 2) == 132 and {c["group"] for c in w} == {"sign", "value", "all"}   # no exponent group
    assert len(s) == 4 * (2 + 21) == 92 and all(c["k"] == 1 for c in s if c["level"] == "layerwise")
    assert any(x.get("group") == "exponent" and "no exponent" in x["reason"] for x in sk)
    assert sum(1 for x in sk if x.get("scope_kind") == "scales" and x.get("k") == 8) == 4
    total = sum(len(S4.enumerate_configs(f, NAMES)[0]) for f in ("e4m11_sn", "fp16", "fp8_e4m3fn", "int8_pt", "int4_pt"))
    assert total == 176 * 3 + 224 * 2 and total * 20 == 19520
    cfg_ns, _ = S4.enumerate_configs("int4_pt", NAMES, include_scales=False)
    assert all(c["scope_kind"] == "weights" for c in cfg_ns)


def test_trial_ids_are_unique_and_deterministic():
    cfg, _ = S4.enumerate_configs("int4_pt", NAMES)
    ids = S4.expected_ids(cfg, 20)
    assert len(ids) == len(set(ids)) == len(cfg) * 20
    assert ids[0] == "int4_pt|weights|GLOBAL|sign|k1|t0"


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    net = build_resnet18_cifar().eval()
    g = torch.Generator().manual_seed(0)
    return net, torch.randn(16, 3, 32, 32, generator=g), torch.randint(0, 10, (16,), generator=g)


def _injector(net, fmt):
    codec = CODECS[fmt]
    enc = OrderedDict((f"{n}.weight", codec.encode(m.weight.detach().numpy())) for n, m in quantizable_modules(net))
    return S3.EncodedFaultInjector(net, S3.EncodedStore(fmt, enc), codec, "cpu")


def test_campaign_resume_completeness_and_summaries(tiny, tmp_path):
    net, X, y = tiny
    fmt = "int4_pt"
    inj = _injector(net, fmt)
    clean = S3.evaluate(inj.model, X, y)
    cfgs, _ = S4.enumerate_configs(fmt, NAMES, layers=[0, 20])
    cfgs = [c for c in cfgs if c["group"] in ("sign", "all") and c["k"] == 8 or c["scope_kind"] == "scales" and c["group"] == "exponent"]
    trials = 3
    path = tmp_path / "fault_trials.jsonl"
    log = S4.TrialLog(path)
    res = S4.run_configs(inj, X, y, clean, CTX, cfgs, trials, 0, log, progress=lambda *_: None)
    log.close()
    assert res["new"] == len(cfgs) * trials and res["skipped_already_done"] == 0
    # resume: nothing new is run and the file does not grow
    size = path.stat().st_size
    log = S4.TrialLog(path)
    res2 = S4.run_configs(inj, X, y, clean, CTX, cfgs, trials, 0, log, progress=lambda *_: None)
    log.close()
    assert res2["new"] == 0 and res2["skipped_already_done"] == len(cfgs) * trials and path.stat().st_size == size
    recs = S4.load_records(path)
    exp = S4.expected_ids(cfgs, trials)
    comp = S4.verify_completeness(recs, exp)
    assert comp["complete"] and comp["recorded"] == len(exp) and not comp["duplicates"]
    for r in recs:
        assert all(f in r for f in S3.REQUIRED_TRIAL_FIELDS) and r["schema"] == S4.SCHEMA_VERSION
        assert r["restored_exactly"] and r["encoded_hash_before"] == r["encoded_hash_after_restore"]
        assert len(r["flips"]) == r["fault_count"] and all(len(f) == 4 for f in r["flips"])
    # interrupted campaign: drop records and detect them; inject a duplicate and detect it
    comp2 = S4.verify_completeness(recs[:-2], exp)
    assert not comp2["complete"] and comp2["n_missing"] == 2
    assert not S4.verify_completeness(recs + [recs[0]], exp)["complete"]
    log = S4.TrialLog(path)
    with pytest.raises(RuntimeError):
        log.append(dict(recs[0]))
    log.close()
    # summaries equal a direct computation
    rows = S4.summarize(recs)
    row = next(r for r in rows if r["layer_name"] == NAMES[20] and r["fault_group"] == "all" and r["k"] == 8 and r["scope_kind"] == "weights")
    rs = [r for r in recs if r["layer_name"] == NAMES[20] and r["fault_group"] == "all" and r["requested_k"] == 8 and r["scope_kind"] == "weights"]
    drops = np.array([r["clean_accuracy"] - r["fault_accuracy"] for r in rs])
    assert row["n_trials"] == trials == len(rs) and row["sample_count"] == 16
    assert row["drop_pp_mean"] == pytest.approx(drops.mean()) and row["drop_pp_worst"] == pytest.approx(drops.max())
    assert row["drop_pp_std"] == pytest.approx(drops.std(ddof=1)) and row["drop_pp_p90"] == pytest.approx(np.percentile(drops, 90))
    assert row["rel_drop_percent_mean"] == pytest.approx(100 * drops.mean() / clean["accuracy"])
    glob, lay = S4.split_global(rows)
    assert glob and lay and all(r["layer_id"] == -1 for r in glob)
    meta = [{"layer_id": i, "module_name": n, "num_weights": 1} for i, n in enumerate(NAMES)]
    header, hm = S4.heatmap(rows, meta)
    assert len(hm) == 21 and any("int4_pt|weights|all|k8|mean_drop_pp" == h for h in header)
    assert hm[20]["int4_pt|weights|all|k8|mean_drop_pp"] != "" and hm[5]["int4_pt|weights|all|k8|mean_drop_pp"] == ""


def test_failures_are_counted_by_cause_and_not_dropped(tiny, tmp_path):
    net, X, y = tiny
    inj = _injector(net, "int8_pt")
    clean = S3.evaluate(inj.model, X, y)
    cfgs = [{"format": "int8_pt", "level": "layerwise", "scope_kind": "scales", "scope_name": NAMES[20], "scope_layers": [20],
             "group": "exponent", "k": 1}]
    log = S4.TrialLog(tmp_path / "t.jsonl")
    S4.run_configs(inj, X, y, clean, CTX, cfgs, 12, 0, log, progress=lambda *_: None)
    log.close()
    recs = S4.load_records(tmp_path / "t.jsonl")
    assert len(recs) == 12
    row = S4.summarize(recs)[0]
    assert row["n_trials"] == 12 and row["failures"] == sum(r["failure"] for r in recs)
    assert row["n_loss_finite"] == sum(r["fault_loss"] is not None for r in recs)
    assert row["failure_rate"] == pytest.approx(row["failures"] / 12)
    assert all(r["fault_accuracy"] is not None for r in recs)              # non-finite trials keep their (declared) accuracy
