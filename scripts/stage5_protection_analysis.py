#!/usr/bin/env python
"""MobileNetV2 protocol Section 5.3: protection analysis (None / Parity / SECDED, plus DMR) on the ACTUAL serialized encoded payload.

Input: the frozen Stage 2 encodings (quantized_weights/<format>.npz, content hashes checked against the quantizer manifest). No model,
no dataset and no GPU are needed: the outcome of a protection code on a word depends on the error pattern, and the payload words are the
real packed bytes (FP16 / BF16 / E4M11-SN words, FP8 bytes, INT8 bytes, INT4 packed nibbles, float32 INT scale).

Definitions (they are also written to fault_protocol.json):
  codeword          one 32-bit logical word of the little-endian serialized payload of one layer; the last word is zero-padded and the
                    padding bits are never fault targets. The float32 INT scale of a layer is one further 32-bit word (region "scale")
                    that is protected in the same way and always tested.
  none / parity / secded / dmr
                    32 data bits | + 1 even-parity bit (33) | extended Hamming (39,32): 6 check bits + overall parity | two full copies.
                    Redundancy lives next to the word (parity placement = bit 32; SECDED check bits at Hamming positions 1,2,4,8,16,32
                    and overall bit 38); redundancy bits are packed densely for the byte accounting.
  scenarios         single_data / double_data (primary, as in the shared protocol: 1 or 2 distinct DATA bits of the word flipped);
                    single_codeword / double_codeword (supplementary, parity and SECDED only: the flips may also hit the stored
                    redundancy bits).
  fairness          None, Parity, SECDED and DMR see the same words and, in the primary scenarios, the same flipped data-bit positions.
  sample            100 codewords per layer and format, drawn without replacement (all of them if the layer has fewer), the same words
                    for every scenario and strategy; seeds are derived from (seed, format, layer, region, word, scenario).
  reaction policy   a word that is detected but not corrected is consumed as read, so its error stays RESIDUAL.

  python scripts/stage5_protection_analysis.py --checkpoint $CK --run-name mobilenetv2_cifar10 --out-dir reports/mobilenetv2_cifar10/stage_05_protection

Writes ecc_validation.json (code self-tests and layout cross-check against the Stage 4 injector), fault_protocol.json,
protection_raw_records.csv.gz (ECC raw records), protection_summary.csv, protection_layer_summary.csv, protection_storage.csv,
protection_storage_summary.csv, protection_report.json. Software fault simulation on stored words, not a hardware reliability measurement.
"""
import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from fr import ecc
from fr import stage2 as S2
from fr import stage3 as S3
from fr.baseline_profile import sha256_file
from fr.config import add_config_args, config_from_args
from fr.model import load_model, quantizable_modules
from fr.utils import dump_json, env_info, git_info

EXPERIMENT = "jfamp-mobilenetv2-cifar10-stage5-protection-analysis"
PRIMARY = ["fp16", "bf16", "e4m11_sn", "fp8_e4m3fn", "fp8_e5m2", "int8_pt", "int4_pt"]
DATA_SCENARIOS = OrderedDict([("single_data", 1), ("double_data", 2)])
CODEWORD_SCENARIOS = OrderedDict([("single_codeword", 1), ("double_codeword", 2)])
CODEWORD_STRATEGIES = ("parity", "secded")
WORDS_PER_LAYER = 100
RAW_FIELDS = ["format_key", "layer_index", "layer_name", "region", "word_index", "payload_byte_offset", "valid_data_bits", "scenario",
              "strategy", "n_faults", "stored_bit_positions", "fault_targets", "original_word_hex", "detected", "corrected", "uncorrectable",
              "miscorrected", "silent", "residual_word_error", "residual_bit_errors", "fault_seed"]
COUNTERS = ["n", "detected", "corrected", "uncorrectable", "miscorrected", "silent", "residual_word", "residual_bits"]
NOMINAL_OVERHEAD = {s: 100.0 * ecc.CHECK_BITS[s] / ecc.WORD_BITS for s in ecc.STRATEGIES}


def seed_of(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()[:8], "little")


# ------------------------------------------------------------------------------------------------ payload layout
def payload_words(enc: dict, width: int):
    """Little-endian 32-bit words of the serialized payload (zero padded) and the number of REAL payload bits in each word."""
    assert sys.byteorder == "little"
    raw = np.ascontiguousarray(enc["words"]).tobytes()
    raw += b"\x00" * ((-len(raw)) % 4)
    words = np.frombuffer(raw, dtype="<u4").astype(np.uint64)
    total = int(enc["n"]) * width                                                  # bits that belong to weights (INT4 padding nibble excluded)
    valid = np.clip(total - 32 * np.arange(words.size), 0, 32)
    return words, valid


def scale_word(enc: dict):
    return np.asarray(enc["scale"], dtype=np.float32).reshape(1).view(np.uint32).astype(np.uint64)


def target_token(fmt: str, width: int, region: str, word: int, bit: int) -> str:
    if region == "scale":
        return f"scale.b{bit}.{S3.field_of_bit(fmt, bit, 'scales')}"
    g = word * 32 + bit
    e, b = divmod(g, width)
    return f"e{e}.b{b}.{S3.field_of_bit(fmt, b)}"


def flip_stream_bit(raw: bytearray, g: int) -> None:
    """Flip bit g of the little-endian bit stream of the payload bytes (what the 32-bit word view addresses as word g // 32, bit g % 32)."""
    w, b = divmod(g, 32)
    raw[w * 4 + b // 8] ^= 1 << (b % 8)


# ------------------------------------------------------------------------------------------------ self-tests
def _t(name, passed, detail=""):
    return {"test": name, "passed": bool(passed), "detail": detail}


def ecc_selftests(seed: int):
    rng = np.random.default_rng(seed)
    out = []
    u = np.uint64
    data = rng.integers(0, 2 ** 32, size=3000, dtype=np.uint64)
    # known answer: data = 1 sits at Hamming position 3 -> check bits at positions 1 and 2, overall parity set
    kat = int(ecc.secded_encode(np.array([1], dtype=np.uint64))[0])
    out.append(_t("secded_known_answer", kat == (1 << 0 | 1 << 1 | 1 << 2 | 1 << 38), f"codeword {kat:#x}"))
    out.append(_t("secded_zero_word", int(ecc.secded_encode(np.zeros(1, dtype=np.uint64))[0]) == 0))
    # independent bit-by-bit reference encoder
    def ref_encode(d):
        bits = {}
        for i, p in enumerate(ecc.DATA_POS):
            bits[p] = (d >> i) & 1
        for c in range(6):
            bits[1 << c] = sum(bits.get(p, 0) for p in range(1, 39) if p & (1 << c) and p != (1 << c)) & 1
        cw = sum(bits[p] << (p - 1) for p in range(1, 39))
        return cw | ((bin(cw).count("1") & 1) << 38)
    out.append(_t("secded_matches_independent_reference_encoder", all(int(c) == ref_encode(int(d)) for c, d in zip(ecc.secded_encode(data[:300]), data[:300]))))
    clean = ecc.outcome("secded", data, np.full((data.size, 1), -1))
    out.append(_t("secded_clean_roundtrip", not clean["detected"].any() and (clean["data_out"] == data).all()))
    ok1 = ok2 = ok3 = True
    d1 = data[:200]
    for p in range(39):                                                           # EVERY single stored bit, incl. check bits
        o = ecc.outcome("secded", d1, np.full((d1.size, 1), p))
        ok1 &= bool(o["corrected"].all() and o["detected"].all() and not o["residual_word"].any() and not o["miscorrected"].any())
    out.append(_t("secded_corrects_every_single_bit_error_of_the_39_stored_bits", ok1, "200 words x 39 positions"))
    pairs = [(a, b) for a in range(39) for b in range(a + 1, 39)]
    d2 = data[:40]
    for a, b in pairs:                                                            # EVERY double error
        o = ecc.outcome("secded", d2, np.tile(np.array([[a, b]]), (d2.size, 1)))
        ok2 &= bool(o["detected"].all() and o["uncorrectable"].all() and not o["corrected"].any() and not o["miscorrected"].any())
    out.append(_t("secded_detects_every_double_bit_error_and_never_miscorrects_it", ok2, f"40 words x {len(pairs)} pairs"))
    d3 = data[:8]
    n_mis = n_det = 0
    for a in range(39):
        for b in range(a + 1, 39):
            for c in range(b + 1, 39):
                o = ecc.outcome("secded", d3, np.tile(np.array([[a, b, c]]), (d3.size, 1)))
                ok3 &= bool(o["detected"].all())                                  # minimum distance 4: a triple is never accepted silently
                n_mis += int(o["miscorrected"].sum() > 0)
                n_det += 1
    out.append(_t("secded_never_accepts_a_triple_error_silently (minimum distance 4)", ok3, f"{n_det} triples, {n_mis} miscorrected as a single"))
    # parity
    okp = True
    for p in range(33):                                                           # every single stored bit (data + parity bit) is detected
        o = ecc.outcome("parity", data[:200], np.full((200, 1), p))
        okp &= bool(o["detected"].all() and not o["corrected"].any())
    out.append(_t("parity_detects_every_single_stored_bit_error_and_corrects_none", okp))
    okd = True
    for a in range(33):
        for b in range(a + 1, 33):
            o = ecc.outcome("parity", data[:5], np.tile(np.array([[a, b]]), (5, 1)))
            okd &= not o["detected"].any()                                        # an even number of flips cancels
    out.append(_t("parity_misses_every_even_weight_pattern (double)", okd))
    # dmr / none
    o = ecc.outcome("dmr", data[:200], np.full((200, 1), 7))
    out.append(_t("dmr_detects_a_flip_in_copy_A_and_returns_it_uncorrected", o["detected"].all() and o["residual_word"].all() and not o["corrected"].any()))
    o = ecc.outcome("dmr", data[:200], np.full((200, 1), 40))
    out.append(_t("dmr_detects_a_flip_in_copy_B_and_data_stays_correct", o["detected"].all() and not o["residual_word"].any()))
    o = ecc.outcome("none", data[:200], np.array([[3, 17]] * 200))
    out.append(_t("none_never_detects_and_keeps_every_flip", (~o["detected"]).all() and (o["residual_bits"] == 2).all() and o["silent"].all()))
    # position mapping of SECDED: the data bit index -> stored position round trip
    ok = True
    for i in range(32):
        o = ecc.outcome("secded", data[:20], np.full((20, 1), ecc.stored_position("secded", i)))
        ok &= bool(o["corrected"].all())
    out.append(_t("secded_data_bit_to_stored_position_map", ok))
    return out


def layout_crosscheck(encoded_by_fmt: dict, seed: int, n: int = 300):
    """The 32-bit word view must address the same physical bit as the Stage 4 injector (``EncodedStore.flip``) for the same
    (layer, element, bit)."""
    rng = np.random.default_rng(seed + 1)
    rows = []
    for fmt, enc in encoded_by_fmt.items():
        width = S3.SPECS[fmt]["width"]
        store = S3.EncodedStore(fmt, enc)
        names = list(enc)
        bad = 0
        for _ in range(n):
            li = int(rng.integers(len(names)))
            e = int(rng.integers(int(store.n[li])))
            b = int(rng.integers(width))
            raw0 = bytearray(np.ascontiguousarray(store.words[li]).tobytes())
            raw1 = bytearray(raw0)
            flip_stream_bit(raw1, e * width + b)                                  # the word-view address of (element, bit)
            idx, old = store.flip(li, e, b)                                       # the Stage 4 injector
            injected = bytes(np.ascontiguousarray(store.words[li]).tobytes())
            store.words[li][idx] = np.asarray(old, dtype=store.words[li].dtype)   # undo
            bad += int(injected != bytes(raw1)) + int(bytes(np.ascontiguousarray(store.words[li]).tobytes()) != bytes(raw0))
        rows.append(_t(f"word_view_matches_stage4_injector[{fmt}]", bad == 0, f"{n} random (layer, element, bit), {bad} mismatches"))
    return rows


# ------------------------------------------------------------------------------------------------ campaign
def sample_words(fmt, lname, region, valid, per_layer, seed):
    pop = np.flatnonzero(valid >= 2)
    if region == "scale" or pop.size <= per_layer:
        return pop
    rng = np.random.default_rng(seed_of(seed, fmt, lname, region, "words"))
    return np.sort(rng.choice(pop, size=per_layer, replace=False))


def process_layer(fmt, li, lname, enc, width, seed, per_layer, emit, acc):
    regions = []
    pw, pv = payload_words(enc, width)
    regions.append(("payload", pw, pv))
    if enc["scale"] is not None:
        regions.append(("scale", scale_word(enc), np.array([32])))
    for region, words, valid in regions:
        idx = sample_words(fmt, lname, region, valid, per_layer, seed)
        data = words[idx]
        v = valid[idx]
        hexes = [f"{int(x):08x}" for x in data]
        # primary scenarios: the same data-bit positions for every strategy
        for sc, k in DATA_SCENARIOS.items():
            seeds = [seed_of(seed, fmt, lname, region, int(w), sc) for w in idx]
            bits = np.array([np.sort(np.random.default_rng(s).choice(int(nv), size=k, replace=False)) for s, nv in zip(seeds, v)])
            for st in ecc.STRATEGIES:
                pos = np.vectorize(lambda b: ecc.stored_position(st, int(b)))(bits)
                o = ecc.outcome(st, data, pos)
                _emit(fmt, li, lname, region, width, idx, v, hexes, sc, st, k, pos, bits, seeds, o, emit, acc)
        # supplementary scenarios: faults anywhere in the stored word of parity / SECDED (incl. the redundancy bits)
        for sc, k in CODEWORD_SCENARIOS.items():
            for st in CODEWORD_STRATEGIES:
                chk = ecc.check_positions(st)
                seeds = [seed_of(seed, fmt, lname, region, int(w), sc, st) for w in idx]
                pos = np.empty((idx.size, k), dtype=np.int64)
                bits = np.full((idx.size, k), -1, dtype=np.int64)
                for r, (s, nv) in enumerate(zip(seeds, v)):
                    cand = [ecc.stored_position(st, b) for b in range(int(nv))] + chk
                    pos[r] = np.sort(np.random.default_rng(s).choice(cand, size=k, replace=False))
                    inv = {ecc.stored_position(st, b): b for b in range(int(nv))}
                    bits[r] = [inv.get(int(p), -1) for p in pos[r]]
                o = ecc.outcome(st, data, pos)
                _emit(fmt, li, lname, region, width, idx, v, hexes, sc, st, k, pos, bits, seeds, o, emit, acc)


def _emit(fmt, li, lname, region, width, idx, v, hexes, sc, st, k, pos, bits, seeds, o, emit, acc):
    key = (fmt, li, region, sc, st)
    a = acc[key]
    a["n"] += idx.size
    for name, col in (("detected", "detected"), ("corrected", "corrected"), ("uncorrectable", "uncorrectable"),
                      ("miscorrected", "miscorrected"), ("silent", "silent"), ("residual_word", "residual_word")):
        a[name] += int(o[col].sum())
    a["residual_bits"] += int(o["residual_bits"].sum())
    for r in range(idx.size):
        toks = [target_token(fmt, width, region, int(idx[r]), int(b)) if b >= 0 else f"check.s{int(p)}" for p, b in zip(pos[r], bits[r])]
        emit([fmt, li, lname, region, int(idx[r]), int(idx[r]) * 4 if region == "payload" else "", int(v[r]), sc, st, k,
              ";".join(str(int(p)) for p in pos[r]), ";".join(toks), hexes[r], int(o["detected"][r]), int(o["corrected"][r]),
              int(o["uncorrectable"][r]), int(o["miscorrected"][r]), int(o["silent"][r]), int(o["residual_word"][r]),
              int(o["residual_bits"][r]), seeds[r]])


def rates(c):
    n = c["n"]
    return {"n_codewords": n, "detection_rate": c["detected"] / n, "correction_rate": c["corrected"] / n,
            "uncorrectable_detected_rate": c["uncorrectable"] / n, "miscorrection_rate": c["miscorrected"] / n,
            "silent_corruption_rate": c["silent"] / n, "residual_word_error_rate": c["residual_word"] / n,
            "mean_residual_bit_errors": c["residual_bits"] / n}


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(ap)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--quant-dir", default=None, help="default results/<run>/jfamp/quantization")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stage4-dir", default=None, help="Stage 4 output directory (only to list the existing fault-campaign artifacts)")
    ap.add_argument("--formats", nargs="+", default=PRIMARY, choices=list(S3.SPECS))
    ap.add_argument("--words-per-layer", type=int, default=WORDS_PER_LAYER)
    ap.add_argument("--fault-seed", type=int, default=42, help="root seed of the codeword sample and the fault positions")
    ap.add_argument("--only-layers", nargs="*", default=None, help="restrict to these module names (smoke runs only)")
    args = ap.parse_args(argv)
    cfg = config_from_args(args)
    base = cfg.results_dir() / "jfamp"
    qdir = Path(args.quant_dir) if args.quant_dir else base / "quantization"
    out = Path(args.out_dir) if args.out_dir else base / "stage_05_protection"
    out.mkdir(parents=True, exist_ok=True)
    ck = Path(args.checkpoint)
    man = json.load(open(qdir / "quantizer_manifest.json"))
    ck_sha = sha256_file(ck)
    if man["checkpoint"]["sha256"] != ck_sha:
        sys.exit("checkpoint SHA-256 differs from the one Stage 2 was frozen on")
    frozen = hashlib.sha256(json.dumps({"checkpoint_sha256": ck_sha, "formats": {k: [v["config"], v["content_sha256"]] for k, v in man["formats"].items()}},
                                       sort_keys=True).encode()).hexdigest()
    if frozen != man["frozen_config_sha256"]:
        sys.exit("Stage 2 manifest does not reproduce its frozen-configuration hash")
    model, _ = load_model(ck)
    module_names = [n for n, _ in quantizable_modules(model)]
    layer_names = [f"{n}.weight" for n in module_names]
    encoded = OrderedDict()
    for k in args.formats:
        if k not in man["formats"]:
            sys.exit(f"format {k} is not in the Stage 2 manifest")
        enc = S2.load_encoded(qdir / man["formats"][k]["encoded_file"], layer_names)
        if S2.content_sha256(enc) != man["formats"][k]["content_sha256"]:
            sys.exit(f"{k}: encoded weights differ from the frozen content hash")
        encoded[k] = enc
    print(f"inputs verified: {len(encoded)} formats x {len(layer_names)} layers, checkpoint {ck_sha[:12]}, frozen config {frozen[:12]}")

    # ---- validation gate for the codes and the word layout
    tests = ecc_selftests(args.fault_seed) + layout_crosscheck(encoded, args.fault_seed)
    for t in tests:
        print(("[PASS] " if t["passed"] else "[FAIL] ") + t["test"] + (f"  ({t['detail']})" if t["detail"] else ""))
    gate = all(t["passed"] for t in tests)
    dump_json({"stage": "JFAMP Stage 5 protection-code validation", "gate": "PASS" if gate else "FAIL", "tests": tests,
               "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen}, out / "ecc_validation.json")
    if not gate:
        sys.exit("protection-code validation failed; no campaign was run")

    # ---- campaign
    acc = defaultdict(lambda: {c: 0 for c in COUNTERS})
    t0, n_rows = time.time(), 0
    storage_rows = []
    only = None if not args.only_layers else set(args.only_layers)
    layer_ids = [(li, mn, ln) for li, (mn, ln) in enumerate(zip(module_names, layer_names)) if only is None or mn in only]
    with gzip.open(out / "protection_raw_records.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=6) as gz:
        w = csv.writer(gz, lineterminator="\n")
        w.writerow(RAW_FIELDS)

        def emit(row):
            nonlocal n_rows
            w.writerow(row)
            n_rows += 1
        for fmt in encoded:
            width = S3.SPECS[fmt]["width"]
            for li, mn, ln in layer_ids:
                enc = encoded[fmt][ln]
                process_layer(fmt, li, mn, enc, width, args.fault_seed, args.words_per_layer, emit, acc)
                st = S2.storage_row(S2.codec_registry()[fmt], enc)
                for s in ecc.STRATEGIES:
                    sb = ecc.storage_bytes(s, st["payload_bytes"], st["scale_bytes"])
                    storage_rows.append({"layer_index": li, "layer_name": mn, "format_key": fmt, "strategy": s, "num_weights": int(enc["n"]),
                                         "payload_bytes": st["payload_bytes"], "scale_bytes": st["scale_bytes"], **sb,
                                         "nominal_overhead_percent": NOMINAL_OVERHEAD[s],
                                         "actual_overhead_percent": 100.0 * (sb["total_bytes"] - sb["unprotected_bytes"]) / sb["unprotected_bytes"]})
            print(f"  {fmt}: done ({time.time() - t0:.0f}s, {n_rows} raw records)", flush=True)

    # ---- summaries
    def pooled(pred):
        c = {k: 0 for k in COUNTERS}
        layers = set()
        for (f, li, region, sc, st), a in acc.items():
            if pred(f, region, sc, st):
                layers.add((f, li))
                for k in COUNTERS:
                    c[k] += a[k]
        return c, len(layers)
    fmt_storage = defaultdict(lambda: {"unprotected": 0, "total": 0})
    for r in storage_rows:
        fmt_storage[(r["format_key"], r["strategy"])]["unprotected"] += r["unprotected_bytes"]
        fmt_storage[(r["format_key"], r["strategy"])]["total"] += r["total_bytes"]
    summary = []
    scs = list(DATA_SCENARIOS) + list(CODEWORD_SCENARIOS)
    for scope in list(encoded) + ["ALL"]:
        for sc in scs:
            for s in ecc.STRATEGIES:
                if sc in CODEWORD_SCENARIOS and s not in CODEWORD_STRATEGIES:
                    continue
                for region in ("payload", "scale"):
                    c, nl = pooled(lambda f, rg, scn, stt: (scope == "ALL" or f == scope) and rg == region and scn == sc and stt == s)
                    if c["n"] == 0:
                        continue
                    if scope == "ALL":
                        un = sum(v["unprotected"] for (f, stt), v in fmt_storage.items() if stt == s)
                        tot = sum(v["total"] for (f, stt), v in fmt_storage.items() if stt == s)
                    else:
                        un, tot = fmt_storage[(scope, s)]["unprotected"], fmt_storage[(scope, s)]["total"]
                    summary.append({"format_key": scope, "scenario": sc, "strategy": s, "region": region, "n_layers": nl, **rates(c),
                                    "nominal_overhead_percent": NOMINAL_OVERHEAD[s], "actual_overhead_percent": 100.0 * (tot - un) / un})
    write_csv(out / "protection_summary.csv", summary)
    layer_rows = []
    for (f, li, region, sc, st), a in sorted(acc.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][3], kv[0][4], kv[0][2])):
        if region == "payload":
            layer_rows.append({"format_key": f, "layer_index": li, "layer_name": module_names[li], "scenario": sc, "strategy": st, **rates(a)})
    write_csv(out / "protection_layer_summary.csv", layer_rows)
    write_csv(out / "protection_storage.csv", storage_rows)
    ss = []
    for (f, s), v in fmt_storage.items():
        ss.append({"format_key": f, "strategy": s, "unprotected_bytes": v["unprotected"], "total_bytes": v["total"],
                   "extra_bytes": v["total"] - v["unprotected"], "nominal_overhead_percent": NOMINAL_OVERHEAD[s],
                   "actual_overhead_percent": 100.0 * (v["total"] - v["unprotected"]) / v["unprotected"]})
    write_csv(out / "protection_storage_summary.csv", ss)

    # ---- completeness: every sampled word has a record in every scenario x strategy
    exp = sum(a["n"] for (f, li, region, sc, st), a in acc.items())
    if exp != n_rows:
        sys.exit(f"record count {n_rows} differs from the accumulated count {exp}")
    s4 = Path(args.stage4_dir) if args.stage4_dir else out.parent / "stage_04_encoded_fault_injection"
    artifacts = OrderedDict([
        ("fault_protocol.json", "stage_05_protection/fault_protocol.json"),
        ("codec_fault_validation.json", "stage_04_encoded_fault_injection/stage4_validation.json (+ known_bit_smoke_tests.csv, no_fault_control.csv)"),
        ("fault_trials.jsonl", "stage_04_encoded_fault_injection/{validation,global,layerwise}_trials.jsonl (+ targeted_trials.jsonl if run)"),
        ("fault_summary.csv", "stage_04_encoded_fault_injection/layerwise_summary.csv"),
        ("global_fault_summary.csv", "stage_04_encoded_fault_injection/global_summary.csv"),
        ("layer_fault_heatmap.csv", "stage_04_encoded_fault_injection/layerwise_heatmap.csv"),
        ("fault_campaign_manifest.json", "stage_04_encoded_fault_injection/{global,layerwise}_manifest.json"),
        ("ECC raw records", "stage_05_protection/protection_raw_records.csv.gz"),
        ("protection_summary.csv", "stage_05_protection/protection_summary.csv")])
    protocol = {
        "stage": "JFAMP protocol Section 5: fault metrics, protection and data collection", "experiment_id": EXPERIMENT,
        "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen, "source_commit": git_info(),
        "formats": list(encoded), "layers": len(layer_names),
        "codeword": {"unit": "one 32-bit logical word of the little-endian serialized encoded payload of one layer",
                     "payload_bits_per_codeword": 32, "partial_word_handling": "last word zero-padded; padding bits are never fault targets; "
                     "alignment padding is counted in the byte overhead", "weights_never_split_across_layers": True,
                     "scales_and_metadata": "the float32 INT scale of a layer (4 bytes) is stored as one more 32-bit word, protected like the payload, "
                     "always tested (region 'scale'); no other metadata exists (codec_metadata_bytes = 0)",
                     "strategies": {"none": {"stored_bits": 32, "redundancy_bits": 0}, "parity": {"stored_bits": 33, "redundancy_bits": 1,
                                    "placement": "even parity of the 32 data bits at bit 32"},
                                    "secded": {"stored_bits": 39, "redundancy_bits": 7, "code": "extended Hamming (39,32)",
                                               "placement": "check bits at Hamming positions 1,2,4,8,16,32 (stored bits 0,1,3,7,15,31), overall parity at stored bit 38"},
                                    "dmr": {"stored_bits": 64, "redundancy_bits": 32, "note": "extension beyond the protocol table; faults hit copy A only; detect-only"}},
                     "redundancy_byte_accounting": "bits packed densely per layer and rounded up to whole bytes; actual vs nominal overhead both reported"},
        "scenarios": {"primary": {k: f"{n} distinct DATA bit(s) of a codeword flipped" for k, n in DATA_SCENARIOS.items()},
                      "supplementary": {k: f"{n} distinct bit(s) anywhere in the stored word (parity and SECDED only: includes the redundancy bits)" for k, n in CODEWORD_SCENARIOS.items()}},
        "sampling": {"codewords_per_layer": args.words_per_layer, "policy": "without replacement from the layer's codewords (all if fewer); the same words for every scenario and strategy; "
                     "same data-bit positions for every strategy in the primary scenarios; per-word seeds derived from sha256(seed|format|layer|region|word|scenario)",
                     "seed": args.fault_seed},
        "reaction_policy": "detected-but-not-corrected words are consumed as read; the error stays residual",
        "metrics": {"detection_rate": "error signalled / codewords", "correction_rate": "data repaired by the code / codewords",
                    "uncorrectable_detected_rate": "signalled and not repaired", "miscorrection_rate": "decoder changed a bit and the data is still wrong",
                    "silent_corruption_rate": "wrong data and no signal", "residual_word_error_rate": "data read back differs from the original",
                    "mean_residual_bit_errors": "differing data bits per codeword"},
        "limits": ["sampled software tests on stored words, not an exhaustive hardware reliability measurement",
                   "model accuracy under faults is NOT re-evaluated here (Stage 4 measured it for the unprotected store)",
                   "the SECDED / parity decoder is not part of an inference kernel; only storage and codeword behaviour are characterized"],
        "required_artifact_map": {k: {"location": v, "exists": True if k in ("fault_protocol.json", "ECC raw records", "protection_summary.csv") else
                                      (s4.exists() and any((s4 / p).exists() for p in ("stage4_validation.json", "layerwise_summary.csv", "global_summary.csv")))}
                                  for k, v in artifacts.items()}}
    dump_json(protocol, out / "fault_protocol.json")
    files = ["ecc_validation.json", "fault_protocol.json", "protection_raw_records.csv.gz", "protection_summary.csv", "protection_layer_summary.csv",
             "protection_storage.csv", "protection_storage_summary.csv"]
    dump_json({"stage": "JFAMP Stage 5 protection analysis", "status": "complete", "experiment_id": EXPERIMENT, "raw_records": n_rows,
               "formats": list(encoded), "layers": len(layer_ids), "codewords_sampled": int(sum(a["n"] for (f, li, rg, sc, st), a in acc.items()
                                                                                                   if sc == "single_data" and st == "none")),
               "words_per_layer": args.words_per_layer, "seed": args.fault_seed, "checkpoint_sha256": ck_sha, "stage2_frozen_config_sha256": frozen,
               "validation_gate": "PASS", "seconds": time.time() - t0, "file_sha256": {f: sha256_file(out / f) for f in files},
               "environment": env_info()}, out / "protection_report.json")
    print(f"written: {out}  ({n_rows} raw records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
