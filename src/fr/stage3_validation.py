"""Validation suites of J-FAMP runbook Stage 3 (Section 5.2). Pure functions returning rows / test dicts.

Every expected value comes from the pure-Python reference decoders of ``stage3`` (first-principles formulas), never from the
codec that is being tested.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from . import stage2 as S2
from . import stage3 as S3

SMOKE_FIELDS = ["test_id", "category", "format", "tensor", "element_index", "field_group", "bit_index", "stored_index",
                "bit_in_stored_unit", "value_before", "word_before_hex", "expected_word_after_hex", "observed_word_after_hex",
                "expected_value_after", "observed_value_after", "expected_change_ratio", "status", "note"]


def _hex(v: int, fmt: str) -> str:
    digits = {16: 4, 8: 2, 4: 1}[S3.SPECS[fmt]["width"]]
    return f"0x{v:0{digits}X}"


def _ref_value(fmt: str, word: int, scale: Optional[float]) -> float:
    sp = S3.SPECS[fmt]
    return S3.ref_decode_float(fmt, word) if sp["kind"] == "float" else S3.ref_decode_int(sp["width"], word, scale)


def synthetic_tensor(fmt: str) -> np.ndarray:
    if fmt == "fp8_e4m3fn":
        return np.array([1.0, -0.75, 0.0625, 0.3, 0.0, 0.0078125], dtype=np.float32)
    if S3.SPECS[fmt]["kind"] == "int":
        return np.array([0.7, -0.7, 0.1, 0.0, 0.35, -0.05, 0.6], dtype=np.float32)      # 7 elements: odd count (INT4 padding)
    return np.array([1.0, -0.75, 0.01, 3.14, 0.25, 0.0009765625], dtype=np.float32)


def _ratio(vb: float, va: float) -> str:
    if vb == 0 or not math.isfinite(vb) or not math.isfinite(va):
        return ""
    return f"{va / vb:.6g}"


def _store_from_words(fmt: str, words: np.ndarray, scale: Optional[float]) -> S3.EncodedStore:
    n = words.size * 2 if fmt == "int4_pt" else words.size
    enc = {"words": words, "scale": None if scale is None else np.float32(scale), "shape": (n,), "n": n}
    return S3.EncodedStore(fmt, {"synthetic": enc})


# ------------------------------------------------------------------------------------------------ bit mapping rows
def bit_mapping_rows(fmt: str, codec: S2.Codec) -> List[dict]:
    """Flip EVERY bit of EVERY element of a tiny synthetic tensor and compare with the reference decoder."""
    sp = S3.SPECS[fmt]
    w = synthetic_tensor(fmt)
    enc = codec.encode(w)
    scale = float(enc["scale"]) if enc["scale"] is not None else None
    rows = []
    base = S3.EncodedStore(fmt, {"synthetic": enc})
    before_words = base.words[0].copy()
    layer_before = S3.decode_layer(codec, base, 0)
    for e in range(w.size):
        for b in range(sp["width"]):
            st = S3.EncodedStore(fmt, {"synthetic": enc})
            wb = st.code(0, e)
            vb = float(S3.decode_elements(codec, st, 0, np.array([e]))[0])
            vb_ref = _ref_value(fmt, wb, scale)
            idx, old = st.flip(0, e, b)
            wa = st.code(0, e)
            exp_wa = wb ^ (1 << b)
            va = float(S3.decode_elements(codec, st, 0, np.array([e]))[0])
            va_ref = _ref_value(fmt, exp_wa, scale)
            diff = np.flatnonzero(before_words != st.words[0])
            xor = int(before_words[diff[0]] ^ st.words[0][diff[0]]) if diff.size == 1 else -1
            exp_idx, exp_pos = (e // 2, 4 * (e % 2) + b) if fmt == "int4_pt" else (e, b)
            others_same = np.array_equal(np.delete(S3.decode_layer(codec, st, 0), e), np.delete(layer_before, e), equal_nan=True)
            ok = (wa == exp_wa and S3.same_value(float(np.float32(va_ref)), va) and S3.same_value(float(np.float32(vb_ref)), vb)
                  and diff.size == 1 and int(diff[0]) == exp_idx and xor == (1 << exp_pos) and idx == exp_idx
                  and S3.bit_location(fmt, e, b) == (exp_idx, exp_pos) and others_same)
            if fmt == "int4_pt":
                ok = ok and S2.int4_bit_location(e, b) == (exp_idx, exp_pos)
            rows.append({"test_id": f"{fmt}:bitmap:e{e}:b{b}", "category": "bit_mapping", "format": fmt, "tensor": "synthetic",
                         "element_index": e, "field_group": S3.field_of_bit(fmt, b), "bit_index": b, "stored_index": idx,
                         "bit_in_stored_unit": exp_pos, "value_before": vb, "word_before_hex": _hex(wb, fmt),
                         "expected_word_after_hex": _hex(exp_wa, fmt), "observed_word_after_hex": _hex(wa, fmt),
                         "expected_value_after": float(np.float32(va_ref)), "observed_value_after": va,
                         "expected_change_ratio": _ratio(vb, va), "status": "PASS" if ok else "FAIL",
                         "note": "only this element changed; stored byte/bit as expected" if ok else "MISMATCH"})
    return rows


# ------------------------------------------------------------------------------------------------ special values
def special_value_rows(fmt: str, codec: S2.Codec) -> List[dict]:
    """Explicit cases at the edges of the encoding: inf/NaN, reserved exponent, subnormal/FTZ, two's complement sign bit."""
    cases = []
    dt = S3.SPECS[fmt]["word_dtype"]
    if fmt == "fp16":
        cases = [("exponent MSB flip of 1.0 (0x3C00) gives +inf", [0x3C00], 14, None),
                 ("exponent MSB flip of 0x3C01 gives NaN", [0x3C01], 14, None)]
    elif fmt == "bf16":
        cases = [("exponent MSB flip of 1.0 (0x3F80) gives +inf (0x7F80)", [0x3F80], 14, None),
                 ("exponent MSB flip of 0x3F81 gives NaN (0x7F81)", [0x3F81], 14, None)]
    elif fmt == "fp8_e5m2":
        cases = [("exponent MSB flip of 1.0 (0x3C) gives +inf (0x7C)", [0x3C], 6, None),
                 ("exponent MSB flip of 0x3D gives NaN (0x7D)", [0x3D], 6, None)]
    elif fmt == "fp8_e4m3fn":
        cases = [("exponent MSB flip of 1.875 (0x3F) gives the single NaN pattern 0x7F", [0x3F], 6, None),
                 ("exponent MSB flip of 1.0 (0x38) gives 256 (0x78), finite", [0x38], 6, None)]
    elif fmt == "e4m11_sn":
        cases = [("exponent MSB flip of 1.0 reaches the reserved es=15, read as es=14 (128)", [0x3800], 14, None),
                 ("es=1 -> es=0 with m=1: subnormal 2^-17", [0x0801], 11, None)]
    elif fmt == "e4m11_ftz":
        cases = [("exponent MSB flip of 1.0 reaches the reserved es=15, read as es=14 (128)", [0x3800], 14, None),
                 ("es=1 -> es=0 with m=1: flushed to zero", [0x0801], 11, None)]
    elif fmt == "int8_pt":
        cases = [("sign bit (bit 7) of code +5 (0x05) gives code 0x85 = -123", [0x05], 7, 0.1),
                 ("sign bit of code -1 (0xFF) gives code +127 (0x7F)", [0xFF], 7, 0.1)]
    elif fmt == "int4_pt":
        cases = [("sign bit (bit 3) of code +7 (0x7) gives code 0xF = -1", [0x7 | 0x70], 3, 0.1),
                 ("sign bit of code -8 (0x8) gives code +0 (0x0)", [0x8 | 0x80], 3, 0.1)]
    rows = []
    for i, (desc, words, bit, scale) in enumerate(cases):
        arr = np.array(words, dtype=dt)
        st = _store_from_words(fmt, arr, scale)
        e = 0
        wb = st.code(0, e)
        vb = float(S3.decode_elements(codec, st, 0, np.array([e]))[0])
        st.flip(0, e, bit)
        wa = st.code(0, e)
        va = float(S3.decode_elements(codec, st, 0, np.array([e]))[0])
        exp_wa = wb ^ (1 << bit)
        va_ref = float(np.float32(_ref_value(fmt, exp_wa, scale)))
        ok = wa == exp_wa and S3.same_value(va_ref, va)
        rows.append({"test_id": f"{fmt}:special:{i}", "category": "special_value", "format": fmt, "tensor": "synthetic",
                     "element_index": e, "field_group": S3.field_of_bit(fmt, bit), "bit_index": bit,
                     "stored_index": 0, "bit_in_stored_unit": bit, "value_before": vb, "word_before_hex": _hex(wb, fmt),
                     "expected_word_after_hex": _hex(exp_wa, fmt), "observed_word_after_hex": _hex(wa, fmt),
                     "expected_value_after": va_ref, "observed_value_after": va, "expected_change_ratio": _ratio(vb, va),
                     "status": "PASS" if ok else "FAIL", "note": desc})
    return rows


# ------------------------------------------------------------------------------------------------ INT scale bits
def scale_bit_rows(fmt: str, codec: S2.Codec) -> List[dict]:
    rows = []
    w = synthetic_tensor(fmt)
    enc = codec.encode(w)
    sc0 = float(enc["scale"])
    u0 = int(np.array(enc["scale"], dtype=np.float32).reshape(1).view(np.uint32)[0])
    for b in range(32):
        st = S3.EncodedStore(fmt, {"synthetic": enc})
        st.flip_scale(0, b)
        with np.errstate(invalid="ignore", over="ignore"):
            got = S3.decode_layer(codec, st, 0).astype(np.float64)
        new_scale = S3.float32_from_bits(u0 ^ (1 << b))
        bits = S3.SPECS[fmt]["width"]
        codes = [st.code(0, e) for e in range(w.size)]
        with np.errstate(invalid="ignore", over="ignore"):
            exp = np.array([float(np.float32(((c - (1 << bits)) if c >= (1 << (bits - 1)) else c)) * np.float32(new_scale))
                            for c in codes])
        ok = all(S3.same_value(float(a), float(c)) for a, c in zip(exp, got))
        rows.append({"test_id": f"{fmt}:scale:b{b}", "category": "scale_bit_mapping", "format": fmt, "tensor": "synthetic scale",
                     "element_index": "ALL", "field_group": S3.field_of_bit(fmt, b, "scales"), "bit_index": b,
                     "stored_index": 0, "bit_in_stored_unit": b, "value_before": sc0,
                     "word_before_hex": f"0x{u0:08X}", "expected_word_after_hex": f"0x{u0 ^ (1 << b):08X}",
                     "observed_word_after_hex": f"0x{int(st.scales[0].reshape(1).view(np.uint32)[0]):08X}",
                     "expected_value_after": new_scale, "observed_value_after": float(st.scales[0]),
                     "expected_change_ratio": "", "status": "PASS" if ok and int(st.scales[0].reshape(1).view(np.uint32)[0]) == (u0 ^ (1 << b)) else "FAIL",
                     "note": "all weights of the tensor rescale by the new scale (float32 bit flip)"})
    return rows


# ------------------------------------------------------------------------------------------------ INT4 packing
def packing_rows(codec: S2.Codec) -> List[dict]:
    """Padding nibble of an odd-length INT4 tensor is never a fault target and never changes."""
    fmt = "int4_pt"
    w = synthetic_tensor(fmt)                       # 7 elements -> 4 bytes, last high nibble is padding
    enc = codec.encode(w)
    rows = []
    st = S3.EncodedStore(fmt, {"synthetic": enc})
    pad_before = int(st.words[0][-1]) >> 4
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(20000):
        f = S3.draw_weight_flips(rng, st, [0], S3.group_bits(fmt, "all"), 1)[0]
        if not 0 <= f[1] < 7:
            bad += 1
        st.flip(0, f[1], f[2])
        if (int(st.words[0][-1]) >> 4) != pad_before:
            bad += 1
        st.flip(0, f[1], f[2])                      # undo
    ok = bad == 0 and pad_before == 0 and S3.EncodedStore(fmt, {"synthetic": enc}).hash() == st.hash()
    rows.append({"test_id": "int4_pt:packing:padding", "category": "packing", "format": fmt, "tensor": "synthetic (7 elements)",
                 "element_index": "0..6", "field_group": "all", "bit_index": "0..3", "stored_index": "0..3",
                 "bit_in_stored_unit": "0..7", "value_before": "", "word_before_hex": f"padding nibble 0x{pad_before:X}",
                 "expected_word_after_hex": "padding nibble stays 0x0", "observed_word_after_hex": "unchanged" if ok else "CHANGED",
                 "expected_value_after": "", "observed_value_after": "", "expected_change_ratio": "",
                 "status": "PASS" if ok else "FAIL",
                 "note": "20,000 random flips: element index always < 7, padding nibble (high nibble of the last byte) never touched"})
    return rows


# ------------------------------------------------------------------------------------------------ synthetic round trip
def roundtrip_synthetic_tests(fmt: str, codec: S2.Codec) -> List[dict]:
    rng = np.random.default_rng(1)
    w = (rng.standard_normal(1001) * 0.02).astype(np.float32)
    e1, e2 = codec.encode(w), codec.encode(w.copy())
    out = [{"test": f"{fmt}: encoding is deterministic (stable bytes)",
            "passed": bool(np.array_equal(e1["words"], e2["words"]))}]
    dec = codec.decode(e1)
    out.append({"test": f"{fmt}: decode returns float32 of the original shape, all finite",
                "passed": bool(dec.dtype == np.float32 and dec.shape == w.shape and np.isfinite(dec).all())})
    if S3.SPECS[fmt]["kind"] == "float":
        out.append({"test": f"{fmt}: encode(decode(words)) reproduces the stored bytes",
                    "passed": bool(np.array_equal(codec.encode(dec)["words"], e1["words"]))})
    else:
        bits = S3.SPECS[fmt]["width"]
        sc = float(e1["scale"])
        codes = (S2.unpack_nibbles(e1["words"], w.size) if bits == 4 else e1["words"]).astype(np.int64)
        ref = np.array([S3.ref_decode_int(bits, int(c), sc) for c in codes], dtype=np.float32)
        out.append({"test": f"{fmt}: decode equals code * float32 scale computed independently",
                    "passed": bool(np.array_equal(ref, dec))})
    return out


# ------------------------------------------------------------------------------------------------ real-tensor round trip
def roundtrip_real_tests(fmt: str, codec: S2.Codec, weights: Mapping[str, np.ndarray], stored: Mapping[str, dict],
                         manifest_hash: str, sample: int = 20000) -> List[dict]:
    """Re-encode the FP32 checkpoint tensors: same bytes as the frozen Stage 2 file; decode of stored words equals the
    independent reference decoder on a random sample of elements."""
    reenc = OrderedDict((n, codec.encode(w)) for n, w in weights.items())
    h_re, h_st = S2.content_sha256(reenc), S2.content_sha256(stored)
    rng = np.random.default_rng(2)
    bad = 0
    total = 0
    store = S3.EncodedStore(fmt, stored)
    for l, n in enumerate(store.names):
        el = rng.integers(0, int(store.n[l]), size=min(sample, int(store.n[l])))
        got = S3.decode_elements(codec, store, l, el)
        sc = float(store.scales[l]) if store.scales[l] is not None else None
        ref = np.array([_ref_value(fmt, store.code(l, int(e)), sc) for e in el[:2000]], dtype=np.float32)
        bad += int(sum(not S3.same_value(float(a), float(b)) for a, b in zip(ref, got[:2000])))
        total += ref.size
        whole = S3.decode_layer(codec, store, l)
        if not np.array_equal(np.nan_to_num(whole, nan=-9), np.nan_to_num(codec.decode(stored[n]), nan=-9)):
            bad += 1
    return [{"test": f"{fmt}: re-encoding the FP32 checkpoint reproduces the stored (frozen) bytes", "passed": h_re == h_st,
             "detail": f"content sha256 {h_re[:16]} vs {h_st[:16]}"},
            {"test": f"{fmt}: stored content hash equals the Stage 2 manifest hash", "passed": h_st == manifest_hash},
            {"test": f"{fmt}: decode of stored words equals the independent reference decoder ({total} sampled elements) "
                     f"and the codec's layer decode", "passed": bad == 0, "detail": f"{bad} mismatches"}]


# ------------------------------------------------------------------------------------------------ scope
def scope_tests(store: S3.EncodedStore, model_param_numels: Mapping[str, int], inventory: Optional[List[dict]],
                fmt: str, n_draws: int = 3000) -> List[dict]:
    out = []
    names_ok = list(model_param_numels) == store.names
    counts_ok = all(int(store.n[i]) == model_param_numels[n] for i, n in enumerate(store.names))
    out.append({"test": f"{fmt}: encoded layer names and element counts equal the model's Conv/Linear weights",
                "passed": names_ok and counts_ok, "detail": f"{len(store.names)} layers, {int(store.n.sum())} elements"})
    if inventory is not None:
        inv_ok = ([f"{r['module_name']}.weight" for r in inventory] == store.names
                  and [int(r["num_weights"]) for r in inventory] == [int(v) for v in store.n])
        out.append({"test": f"{fmt}: layer order, names and counts equal the Stage 1 inventory (stable layer key)", "passed": inv_ok})
    rng = np.random.default_rng(3)
    bad = 0
    for l in range(len(store.names)):                                       # layer-wise scope
        for _ in range(n_draws // len(store.names) + 1):
            for f in S3.draw_weight_flips(rng, store, [l], S3.group_bits(fmt, "all"), 4):
                bad += int(f[0] != l or not 0 <= f[1] < int(store.n[l]))
    allf = []
    for _ in range(n_draws):                                                # global scope
        fl = S3.draw_weight_flips(rng, store, list(range(len(store.names))), S3.group_bits(fmt, "all"), 8)
        bad += int(len({(a, b) for a, b, _ in fl}) != 8)                    # k distinct elements
        bad += int(sum(not 0 <= e < int(store.n[l]) for l, e, _ in fl))
        allf += [f[0] for f in fl]
    # uniform over WEIGHTS: the share of flips per layer should follow the layer size
    share = np.bincount(allf, minlength=len(store.names)) / len(allf)
    expect = store.n / store.n.sum()
    chi = float(np.abs(share - expect).max())
    out.append({"test": f"{fmt}: drawn locations lie inside the declared scope and layer; k=8 flips hit 8 distinct elements",
                "passed": bad == 0, "detail": f"{bad} violations"})
    out.append({"test": f"{fmt}: global scope is uniform over weights (per-layer share of flips follows layer size)",
                "passed": chi < 0.02, "detail": f"max |share - size share| = {chi:.4f}"})
    return out


# ------------------------------------------------------------------------------------------------ model-level suite
def model_level_tests(fmt: str, inj: S3.EncodedFaultInjector, X, y, ctx: dict, base_seed: int, fp32_hash_fn,
                      n_restore_trials: int = 12) -> dict:
    """No-fault control, restoration, forced non-finite handling on the real model. Returns {"tests": [...], "records": [...]}."""
    store = inj.store
    L = list(range(len(store.names)))
    tests, records = [], []
    fp32_before = fp32_hash_fn()
    h_start = store.hash()
    snap = [f.clone() for f in inj.flat]
    clean = S3.evaluate(inj.model, X, y, return_logits=True)
    clean_logits = clean.pop("logits")
    clean_rec = {"accuracy": clean["accuracy"], "loss": clean["loss"]}

    # ---- no-fault control (zero flips through the full trial machinery)
    r0 = S3.run_trial(inj, X, y, clean_rec, ctx, fmt=fmt, scope_layers=L, scope_name="GLOBAL", scope_kind="weights",
                      group="none", k=0, trial=0, base_seed=base_seed, return_logits=True)
    lg0 = r0.pop("_logits")
    tests.append({"test": f"{fmt}: no-fault control (0 flips) reproduces the clean quantized model exactly (logits bit-equal)",
                  "passed": bool(torch.equal(lg0, clean_logits)) and r0["fault_accuracy"] == clean["accuracy"] and r0["restored_exactly"]
                            and r0["encoded_hash_faulted"] == r0["encoded_hash_before"],
                  "detail": f"clean accuracy {clean['accuracy']:.2f} % on {int(y.numel())} images"})
    records.append(r0)

    # ---- restoration after trials of different groups / k / scopes (hash before == hash after, weights bit-equal)
    groups = [g for g in S3.format_groups(fmt) if S3.group_bits(fmt, g)]
    cases = []
    for t in range(n_restore_trials):
        g = groups[t % len(groups)]
        k = (1, 8)[t % 2]
        cases.append((g, k, t))
    all_ok, applied_ok = True, True
    for g, k, t in cases:
        r = S3.run_trial(inj, X, y, clean_rec, ctx, fmt=fmt, scope_layers=L, scope_name="GLOBAL", scope_kind="weights", group=g,
                         k=k, trial=100 + t, base_seed=base_seed)
        records.append(r)
        all_ok &= r["restored_exactly"]
        applied_ok &= r["flip_applied_to_encoded_bytes"] and r["encoded_hash_faulted"] != r["encoded_hash_before"]
    lg_after = S3.evaluate(inj.model, X, y, return_logits=True)["logits"]
    weights_equal = all(torch.equal(a, b) for a, b in zip(snap, inj.flat))
    tests.append({"test": f"{fmt}: restoration - encoded-state hash before == after for all {len(cases)} trials (groups {groups}, k 1 and 8)",
                  "passed": all_ok, "detail": f"stored state sha256 {h_start[:16]}"})
    tests.append({"test": f"{fmt}: every faulted trial changed the ENCODED bytes (hash during the fault differs from before)", "passed": applied_ok})
    tests.append({"test": f"{fmt}: after the trials the decoded weights are bit-identical to the clean ones and logits equal the clean logits",
                  "passed": bool(weights_equal and torch.equal(lg_after, clean_logits) and store.hash() == h_start)})
    tests.append({"test": f"{fmt}: the FP32 source checkpoint tensors were not touched by any injection",
                  "passed": fp32_hash_fn() == fp32_before})

    # ---- forced non-finite scenarios: detection in weights / activations / logits / loss, trial kept and flagged
    forced = _forced_nonfinite_case(fmt, store)
    if forced is not None:
        kind, wf, sf, expect_nonfinite, desc = forced
        r = S3.run_trial(inj, X, y, clean_rec, ctx, fmt=fmt, scope_layers=L, scope_name="GLOBAL",
                         scope_kind="scales" if sf else "weights", group="forced", k=len(wf) + len(sf), trial=900,
                         base_seed=base_seed, explicit_weight_flips=wf, explicit_scale_flips=sf)
        records.append(r)
        flags = r["output_flag"].split(",")
        if expect_nonfinite:
            ok = (r["nonfinite_weight_count"] > 0 and all(f in flags for f in
                  ("weights_nonfinite", "activations_nonfinite", "logits_nonfinite", "loss_nonfinite")) and r["fault_loss"] is None
                  and r["failure"])
        else:
            ok = r["nonfinite_weight_count"] == 0 and r["output_flag"] == "none" and r["fault_loss"] is not None
        lg = S3.evaluate(inj.model, X, y, return_logits=True)["logits"]
        tests.append({"test": f"{fmt}: forced scenario - {desc}", "passed": bool(ok and r["restored_exactly"] and torch.equal(lg, clean_logits)),
                      "detail": f"nonfinite weights {r['nonfinite_weight_count']}, flags {r['output_flag']}, fault accuracy "
                                f"{r['fault_accuracy']:.2f} %, fault loss {r['fault_loss']}; trial recorded and state restored"})
    return {"tests": tests, "records": records, "clean": clean_rec}


def _forced_nonfinite_case(fmt: str, store: S3.EncodedStore):
    """(kind, weight_flips, scale_flips, expect_nonfinite, description) built from several flips of ONE element/scale."""
    l = len(store.names) - 1                                   # fc layer
    e = 0
    if fmt in ("fp16", "bf16", "fp8_e5m2"):
        w = store.code(l, e)
        bits = [b for b in S3.SPECS[fmt]["fields"]["exponent"] if not (w >> b) & 1]
        return "w", [(l, e, b) for b in bits], [], True, "exponent forced to all ones (inf/NaN) in one fc weight"
    if fmt == "fp8_e4m3fn":
        w = store.code(l, e)
        bits = [b for b in range(7) if not (w >> b) & 1]
        return "w", [(l, e, b) for b in bits], [], True, "one fc weight forced to the NaN pattern S.1111.111"
    if fmt in ("int8_pt", "int4_pt"):
        u = int(store.scales[l].reshape(1).view(np.uint32)[0])
        bits = [b for b in S3.SCALE_FIELDS["exponent"] if not (u >> b) & 1]
        return "s", [], [(l, b) for b in bits], True, "float32 scale of the fc layer forced to exponent all ones (inf/NaN)"
    if fmt in ("e4m11_sn", "e4m11_ftz"):
        w = store.code(l, e)
        bits = [b for b in S3.SPECS[fmt]["fields"]["exponent"] if not (w >> b) & 1]
        return "w", [(l, e, b) for b in bits], [], False, \
            "reserved exponent es=15 is read as es=14 -> weight stays finite, no non-finite value can occur"
    return None
