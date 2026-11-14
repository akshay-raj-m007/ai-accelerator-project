import csv
import gzip
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("stage5prot", ROOT / "scripts" / "stage5_protection_analysis.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)

from fr import ecc  # noqa: E402
from fr import stage2 as S2  # noqa: E402
from fr.baseline_profile import sha256_file  # noqa: E402
from fr.model import build_mobilenetv2_cifar, quantizable_modules  # noqa: E402

FORMATS = ["fp16", "bf16", "e4m11_sn", "fp8_e4m3fn", "fp8_e5m2", "int8_pt", "int4_pt"]


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    model = build_mobilenetv2_cifar(10).eval()
    mods = quantizable_modules(model)
    return model, mods


def _encode_all(mods, fmts):
    reg = S2.codec_registry()
    return OrderedDict((k, OrderedDict((f"{n}.weight", reg[k].encode(m.weight.detach().numpy())) for n, m in mods)) for k in fmts)


def test_code_selftests_pass():
    res = M.ecc_selftests(42)
    assert len(res) == 13 and all(t["passed"] for t in res), [t for t in res if not t["passed"]]


def test_secded_layout_has_32_data_and_7_check_bits_and_known_masks():
    assert ecc.CHECK_POS == (1, 2, 4, 8, 16, 32) and len(ecc.DATA_POS) == 32 and max(ecc.DATA_POS) == 38
    assert ecc.STORED_BITS["secded"] == 39 and ecc.CHECK_BITS["secded"] == 7
    assert ecc.stored_position("secded", 0) == 2 and ecc.stored_position("parity", 5) == 5      # data bit 0 sits at Hamming position 3


def test_secded_is_a_linear_distance_4_code_on_random_words():
    rng = np.random.default_rng(1)
    d = rng.integers(0, 2 ** 32, size=500, dtype=np.uint64)
    e = ecc.secded_encode(d)
    # linearity: enc(a) ^ enc(b) == enc(a ^ b)
    assert (ecc.secded_encode(d[:250] ^ d[250:]) == (e[:250] ^ e[250:])).all()
    # every non-zero codeword of the sample has weight >= 4
    assert ecc.popcount64(e[d != 0]).min() >= 4


@pytest.mark.parametrize("strategy,nominal", [("none", 0.0), ("parity", 3.125), ("secded", 21.875), ("dmr", 100.0)])
def test_nominal_overhead(strategy, nominal):
    assert M.NOMINAL_OVERHEAD[strategy] == nominal


def test_storage_accounting_counts_alignment_padding_scale_word_and_packed_redundancy():
    s = ecc.storage_bytes("secded", 433, 4)             # 433 payload bytes -> 109 words, +1 scale word = 110 words
    assert s["alignment_padding_bytes"] == 3 and s["protected_words"] == 110
    assert s["redundancy_bytes"] == -(-110 * 7 // 8) == 97
    assert s["total_bytes"] == 433 + 4 + 3 + 97
    p = ecc.storage_bytes("parity", 432, 0)
    assert p["redundancy_bytes"] == 14 and p["total_bytes"] == 432 + 14               # 108 words -> 108 bits -> 14 bytes
    d = ecc.storage_bytes("dmr", 432, 4)
    assert d["total_bytes"] == 2 * 436 and ecc.storage_bytes("none", 432, 4)["total_bytes"] == 436


def test_word_view_addresses_the_same_bit_as_the_stage4_injector(tiny):
    _, mods = tiny
    enc = _encode_all(mods, FORMATS)
    res = M.layout_crosscheck(enc, 42, n=120)
    assert len(res) == 7 and all(t["passed"] for t in res), res


def test_payload_words_are_little_endian_with_zero_padding_and_valid_bit_counts(tiny):
    enc = {"words": np.array([0x0201, 0x0403, 0x0005], dtype=np.uint16), "n": 3}
    w, v = M.payload_words(enc, 16)
    assert w.tolist() == [0x04030201, 0x00000005] and v.tolist() == [32, 16]
    e4 = {"words": np.array([0x21, 0x43, 0x05], dtype=np.uint8), "n": 5}          # 5 INT4 elements -> 20 real bits; the 6th nibble is padding
    w, v = M.payload_words(e4, 4)
    assert w.tolist() == [0x00054321] and v.tolist() == [20]


def _run_layer(enc, width, fmt="fp16", per_layer=100):
    rows, acc = [], defaultdict(lambda: {c: 0 for c in M.COUNTERS})
    M.process_layer(fmt, 0, "L", enc, width, 7, per_layer, rows.append, acc)
    return rows, acc


def test_process_layer_reproduces_the_expected_protection_behaviour(tiny):
    _, mods = tiny
    name, m = mods[5]                                                                  # a pointwise layer, > 100 words
    enc = S2.codec_registry()["fp16"].encode(m.weight.detach().numpy())
    rows, acc = _run_layer(enc, 16)
    g = lambda sc, st, k: acc[("fp16", 0, "payload", sc, st)][k]
    n = g("single_data", "none", "n")
    assert n == 100
    for st in ecc.STRATEGIES:                                                          # same words everywhere
        assert g("single_data", st, "n") == g("double_data", st, "n") == n
    assert g("single_data", "none", "silent") == n and g("single_data", "none", "detected") == 0
    assert g("single_data", "parity", "detected") == n and g("single_data", "parity", "corrected") == 0 and g("single_data", "parity", "residual_word") == n
    assert g("double_data", "parity", "detected") == 0 and g("double_data", "parity", "silent") == n
    assert g("single_data", "secded", "corrected") == n and g("single_data", "secded", "residual_word") == 0
    assert g("double_data", "secded", "uncorrectable") == n and g("double_data", "secded", "corrected") == 0 and g("double_data", "secded", "miscorrected") == 0
    assert g("double_data", "secded", "residual_word") == n and g("double_data", "secded", "residual_bits") == 2 * n
    assert g("single_data", "dmr", "detected") == n and g("single_data", "dmr", "corrected") == 0 and g("single_data", "dmr", "residual_word") == n
    # supplementary scenarios exist for parity and SECDED only; faults landing on check bits leave the data intact
    assert ("fp16", 0, "payload", "single_codeword", "none") not in acc
    assert g("single_codeword", "secded", "detected") == n and g("single_codeword", "secded", "residual_word") == 0
    assert g("single_codeword", "parity", "detected") == n
    assert 0 < g("double_codeword", "parity", "residual_word") <= n
    # one raw record per sampled word x scenario x strategy (4 x 4 primary combos... 2 scenarios x 4 + 2 x 2 supplementary = 12)
    assert len(rows) == 100 * 12 and len(rows[0]) == len(M.RAW_FIELDS)
    # fault_targets name a real element, bit and field
    tok = [r[M.RAW_FIELDS.index("fault_targets")] for r in rows if r[M.RAW_FIELDS.index("scenario")] == "single_data"][0]
    assert tok.startswith("e") and tok.split(".")[-1] in ("sign", "exponent", "fraction")


def test_the_same_data_bits_are_flipped_for_every_strategy(tiny):
    _, mods = tiny
    enc = S2.codec_registry()["fp8_e4m3fn"].encode(mods[3][1].weight.detach().numpy())
    rows, _ = _run_layer(enc, 8, "fp8_e4m3fn")
    f = M.RAW_FIELDS
    by = defaultdict(set)
    for r in rows:
        if r[f.index("scenario")] in ("single_data", "double_data"):
            by[(r[f.index("scenario")], r[f.index("word_index")])].add(r[f.index("fault_targets")])
    assert by and all(len(v) == 1 for v in by.values())


def test_small_layer_uses_all_its_words_and_int_scale_word_is_always_tested(tiny):
    _, mods = tiny
    name, m = mods[1]                                                                 # features.1.conv.0.0 depthwise, 288 weights
    enc = S2.codec_registry()["int4_pt"].encode(m.weight.detach().numpy())
    rows, acc = _run_layer(enc, 4, "int4_pt")
    assert acc[("int4_pt", 0, "payload", "single_data", "none")]["n"] == 36            # 144 bytes = 36 words < 100
    assert acc[("int4_pt", 0, "scale", "single_data", "secded")]["n"] == 1
    sc = [r for r in rows if r[M.RAW_FIELDS.index("region")] == "scale" and r[M.RAW_FIELDS.index("scenario")] == "single_data"]
    assert sc and sc[0][M.RAW_FIELDS.index("fault_targets")].startswith("scale.b")


def test_end_to_end_script_writes_every_artifact_and_is_reproducible(tmp_path, tiny, monkeypatch):
    model, mods = tiny
    ck = tmp_path / "ck.pth"
    torch.save({"epoch": 1, "model_state_dict": model.state_dict(),
                "config": {"arch": "mobilenetv2_cifar", "val_size": 5000, "seed": 0, "normalization": {"mean": [0.5] * 3, "std": [0.2] * 3}}}, ck)
    fmts = ["fp16", "int4_pt"]
    enc = _encode_all(mods, fmts)
    q = tmp_path / "quant"
    q.mkdir()
    fm = {}
    for k in fmts:
        S2.save_encoded(q / f"{k}.npz", enc[k])
        fm[k] = {"config": S2.codec_registry()[k].config(), "content_sha256": S2.content_sha256(enc[k]), "encoded_file": f"{k}.npz", "role": "primary"}
    sha = sha256_file(ck)
    fr = hashlib.sha256(json.dumps({"checkpoint_sha256": sha, "formats": {k: [v["config"], v["content_sha256"]] for k, v in fm.items()}}, sort_keys=True).encode()).hexdigest()
    json.dump({"checkpoint": {"sha256": sha}, "formats": fm, "frozen_config_sha256": fr}, open(q / "quantizer_manifest.json", "w"))
    outs = []
    for tag in ("a", "b"):
        out = tmp_path / tag
        assert M.main(["--checkpoint", str(ck), "--quant-dir", str(q), "--out-dir", str(out), "--device", "cpu", "--formats", *fmts,
                       "--only-layers", mods[1][0], mods[5][0], mods[52][0]]) == 0
        outs.append(out)
    names = ["ecc_validation.json", "fault_protocol.json", "protection_raw_records.csv.gz", "protection_summary.csv", "protection_layer_summary.csv",
             "protection_storage.csv", "protection_storage_summary.csv", "protection_report.json"]
    assert all((outs[0] / n).exists() for n in names)
    assert json.load(open(outs[0] / "ecc_validation.json"))["gate"] == "PASS"
    rep = json.load(open(outs[0] / "protection_report.json"))
    assert rep["status"] == "complete" and rep["layers"] == 3
    with gzip.open(outs[0] / "protection_raw_records.csv.gz", "rt", encoding="utf-8") as fh:
        raw = list(csv.DictReader(fh))
    assert len(raw) == rep["raw_records"] and list(raw[0]) == M.RAW_FIELDS
    summ = {(r["format_key"], r["scenario"], r["strategy"], r["region"]): r for r in csv.DictReader(open(outs[0] / "protection_summary.csv"))}
    s = summ[("ALL", "single_data", "secded", "payload")]
    assert float(s["correction_rate"]) == 1.0 and float(s["residual_word_error_rate"]) == 0.0 and float(s["nominal_overhead_percent"]) == 21.875
    assert float(summ[("ALL", "double_data", "parity", "payload")]["detection_rate"]) == 0.0
    assert float(summ[("fp16", "single_data", "secded", "payload")]["actual_overhead_percent"]) >= 21.875
    # no scale word for FP16
    assert ("fp16", "single_data", "none", "scale") not in summ and ("int4_pt", "single_data", "none", "scale") in summ
    # the sampled-word raw records are identical across two runs (seeded)
    for n in ("protection_summary.csv", "protection_layer_summary.csv", "protection_storage.csv"):
        assert (outs[0] / n).read_bytes() == (outs[1] / n).read_bytes()
    with gzip.open(outs[0] / "protection_raw_records.csv.gz", "rb") as a, gzip.open(outs[1] / "protection_raw_records.csv.gz", "rb") as b:
        assert a.read() == b.read()
    proto = json.load(open(outs[0] / "fault_protocol.json"))
    assert proto["codeword"]["payload_bits_per_codeword"] == 32 and proto["sampling"]["codewords_per_layer"] == 100


def test_a_changed_encoding_is_refused(tmp_path, tiny):
    model, mods = tiny
    ck = tmp_path / "ck.pth"
    torch.save({"epoch": 1, "model_state_dict": model.state_dict(),
                "config": {"arch": "mobilenetv2_cifar", "val_size": 5000, "seed": 0, "normalization": {"mean": [0.5] * 3, "std": [0.2] * 3}}}, ck)
    enc = _encode_all(mods, ["fp16"])["fp16"]
    q = tmp_path / "quant"
    q.mkdir()
    S2.save_encoded(q / "fp16.npz", enc)
    fm = {"fp16": {"config": S2.codec_registry()["fp16"].config(), "content_sha256": "0" * 64, "encoded_file": "fp16.npz", "role": "primary"}}
    sha = sha256_file(ck)
    fr = hashlib.sha256(json.dumps({"checkpoint_sha256": sha, "formats": {k: [v["config"], v["content_sha256"]] for k, v in fm.items()}}, sort_keys=True).encode()).hexdigest()
    json.dump({"checkpoint": {"sha256": sha}, "formats": fm, "frozen_config_sha256": fr}, open(q / "quantizer_manifest.json", "w"))
    with pytest.raises(SystemExit, match="frozen content hash"):
        M.main(["--checkpoint", str(ck), "--quant-dir", str(q), "--out-dir", str(tmp_path / "o"), "--device", "cpu", "--formats", "fp16"])
