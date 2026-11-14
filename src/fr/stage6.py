"""MobileNetV2 protocol Section 6: joint datatype-protection optimisation (candidate table, constrained assignment, protected end-to-end faults).

Candidate = (layer l, datatype d, protection p). Datatypes: the 7 primary formats. Protections (all implemented and measured):
  none      unprotected store
  parity    one even-parity bit per 32-bit word (Stage 5; detect-only, a detected word is consumed as read)
  secded    extended Hamming (39,32) per 32-bit word (Stage 5)
  dmr       two copies (Stage 5; detect-only)
  scale_tmr INT formats only: the float32 scale word stored three times, read by bit-wise majority (fr.protect, as in the ResNet-18 run); payload unprotected
28 word-level options + 2 scale_tmr options = 30 options per layer, 53 x 30 = 1,590 candidates. Unsupported combinations (scale_tmr on a float format) do not exist.

  Q(l,d)    one-layer-at-a-time clean penalty: max(0, CE change on the 5,000 validation images when ONLY layer l is stored in d)   (Stage 3, the empirical
            proxy the protocol prefers). No fault damage in Q.
  R(l,d,p)  residual fault risk (pp of validation accuracy) under the DECLARED distribution: k = 8 distinct weights drawn over all quantizable weights of the model,
            one uniformly drawn bit of each weight's own encoding. R_exposure(l,d,none) = 8 (n_l / N) * per_flip(l,d), per_flip = sum over fault groups g of
            (bits_g / width) * max(0, mean(drop_k1_g, drop_k8_g / 8))   (Stage 4 layer-wise group results, 10 trials x 1,000 validation images each); additive
            over layers. R(l,d,p) = R(l,d,none) * residual_rate(l,d,p), where residual_rate is the MEASURED fraction of single-bit data faults whose error is still
            present after the protection's decoder (Stage 5 protection_layer_summary: parity 1, secded 0, dmr 1; scale_tmr leaves the payload unprotected: 1).
            This is a derived estimate; the assembled model is then measured end to end through the real decoders.
  R_scale   SECOND metric, never merged into R: mean accuracy drop of ONE flipped bit of the layer's float32 scale word (all 32 bits, exhaustive,
            stage6_scale_faults.py) times the measured residual rate of the protection on the scale word (parity 1, secded 0, dmr 1; scale_tmr 0 by construction,
            checked exhaustively in tests). Options with R_scale above tau_scale are inadmissible.
  S(l,d,p)  actual stored bytes: payload + scale word(s) + 32-bit alignment padding + packed redundancy bits (Stage 5 protection_storage.csv); scale_tmr: + 8 bytes.
  Normalisation: global min-max over all candidates (as the ResNet-18 optimiser), lambda = 1.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
from collections import Counter, OrderedDict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import ecc
from . import protect as P
from . import stage3 as S3
from . import stage5 as S5

FORMATS = ["fp16", "bf16", "e4m11_sn", "fp8_e4m3fn", "fp8_e5m2", "int8_pt", "int4_pt"]
WORD_PROTECTIONS = ("none", "parity", "secded", "dmr")
TMR = P.PROTECTION_TMR
SHORT = {"fp16": "FP16", "bf16": "BF16", "e4m11_sn": "E4M11", "fp8_e4m3fn": "FP8N", "fp8_e5m2": "FP8E5", "int8_pt": "INT8", "int4_pt": "INT4"}
PSHORT = {"none": "", "parity": "+P", "secded": "+S", "dmr": "+D", TMR: "+T"}
K_FAULTS = 8
DISTRIBUTIONS = ("weights_k8", "stored_k8", "scales_k1")
DIST_K = {"weights_k8": 8, "stored_k8": 8, "scales_k1": 1}
SCHEMA = "mnv2-stage6-e2e-trial-v1"


# ------------------------------------------------------------------------------------------------ options
def option_token(fmt: str, prot: str) -> str:
    return fmt if prot == "none" else f"{fmt}+{prot}"


def parse_option(token: str) -> Tuple[str, str]:
    fmt, _, prot = token.partition("+")
    prot = prot or "none"
    if fmt not in FORMATS:
        raise ValueError(f"unknown datatype {fmt!r}")
    if prot not in WORD_PROTECTIONS + (TMR,):
        raise ValueError(f"unknown protection {prot!r}")
    if prot == TMR and S3.SPECS[fmt]["kind"] != "int":
        raise ValueError(f"{TMR} needs an INT format (it protects the scale word)")
    return fmt, prot


def short_token(token: str) -> str:
    f, p = parse_option(token)
    return SHORT[f] + PSHORT[p]


def long_token(short: str) -> str:
    for f in FORMATS:
        for p in WORD_PROTECTIONS + (TMR,):
            if short == SHORT[f] + PSHORT[p]:
                return option_token(f, p)
    raise ValueError(short)


def all_options() -> List[str]:
    out = [option_token(f, p) for f in FORMATS for p in WORD_PROTECTIONS]
    out += [option_token(f, TMR) for f in FORMATS if S3.SPECS[f]["kind"] == "int"]
    return out


def base_token(token: str) -> str:
    """Token understood by the ResNet-era MixedStore: word-level protections are applied outside the store (see ProtectedFaults)."""
    f, p = parse_option(token)
    return option_token(f, TMR) if p == TMR else f


# ------------------------------------------------------------------------------------------------ candidate table
def _f(x) -> float:
    return float(x)


def per_flip_drop(fmt: str, s4: Dict[tuple, dict], layer: str) -> Tuple[float, dict]:
    """sum_g (bits_g / width) * max(0, mean(k1 drop, k8 drop / 8)) from the Stage 4 layer-wise group rows (bit uniform over all bits of the encoding)."""
    sp = S3.SPECS[fmt]
    total, parts = 0.0, {}
    for g, bits in sp["fields"].items():
        k1, k8 = _f(s4[(layer, fmt, g, 1)]["drop_pp_mean"]), _f(s4[(layer, fmt, g, 8)]["drop_pp_mean"])
        pf = S5.per_flip_drop(k1, k8)
        parts[g] = {"weight": len(bits) / sp["width"], "k1_drop_pp": k1, "k8_drop_pp": k8, "per_flip_pp": pf}
        total += len(bits) / sp["width"] * pf
    return total, parts


def build_candidates(s3_rows: Sequence[dict], s4_rows: Sequence[dict], storage_rows: Sequence[dict], prot_layer_rows: Sequence[dict],
                     prot_summary_rows: Sequence[dict], scale_rows: Sequence[dict]) -> Tuple[List[dict], dict]:
    """The 1,590-row candidate table from the Stage 3 / 4 / 5 outputs and the scale-word campaign (all plain CSV rows)."""
    q3 = {(r["layer_name"], r["format_key"]): r for r in s3_rows}
    s4 = {}
    for r in s4_rows:
        if r["scope_kind"] == "weights":
            s4[(r["layer_name"], r["format"], r["fault_group"], int(r["k"]))] = r
    stor = {(r["layer_name"], r["format_key"], r["strategy"]): r for r in storage_rows}
    resid = {(r["layer_name"], r["format_key"], r["strategy"]): _f(r["residual_word_error_rate"]) for r in prot_layer_rows if r["scenario"] == "single_data"}
    scale_resid = {(r["format_key"], r["strategy"]): _f(r["residual_word_error_rate"]) for r in prot_summary_rows
                   if r["scenario"] == "single_data" and r["region"] == "scale" and r["format_key"] != "ALL"}
    scale_drop = {(r["layer_name"], r["format_key"]): _f(r["mean_drop_pp"]) for r in scale_rows}
    layers = OrderedDict()
    for r in sorted(storage_rows, key=lambda r: int(r["layer_index"])):
        layers.setdefault(r["layer_name"], {"layer_index": int(r["layer_index"]), "num_weights": int(r["num_weights"])})
    N = sum(v["num_weights"] for v in layers.values())
    rows = []
    for lname, info in layers.items():
        n_l = info["num_weights"]
        for fmt in FORMATS:
            q = q3[(lname, fmt)]
            Q = max(0.0, _f(q["loss_change"]))
            pf, parts = per_flip_drop(fmt, s4, lname)
            R_none = K_FAULTS * (n_l / N) * pf
            is_int = S3.SPECS[fmt]["kind"] == "int"
            Rs_none = max(0.0, scale_drop[(lname, fmt)]) if is_int else 0.0
            for prot in WORD_PROTECTIONS + ((TMR,) if is_int else ()):
                if prot == TMR:
                    f_w, f_s = 1.0, 0.0
                    sr = stor[(lname, fmt, "none")]
                    S = int(sr["unprotected_bytes"]) + P.TMR_OVERHEAD_BYTES
                    parts_b = {"payload_bytes": int(sr["payload_bytes"]), "scale_bytes": int(sr["scale_bytes"]), "alignment_padding_bytes": 0,
                               "redundancy_bytes": P.TMR_OVERHEAD_BYTES}
                else:
                    f_w = 1.0 if prot == "none" else resid[(lname, fmt, prot)]
                    f_s = 1.0 if prot == "none" else (scale_resid[(fmt, prot)] if is_int else 0.0)
                    sr = stor[(lname, fmt, prot)]
                    S = int(sr["total_bytes"])
                    parts_b = {"payload_bytes": int(sr["payload_bytes"]), "scale_bytes": int(sr["scale_bytes"]),
                               "alignment_padding_bytes": int(sr["alignment_padding_bytes"]), "redundancy_bytes": int(sr["redundancy_bytes"])}
                rows.append({"layer_index": info["layer_index"], "layer_name": lname, "layer_type": q["layer_type"], "num_weights": n_l, "datatype": fmt, "protection": prot,
                             "option": option_token(fmt, prot), "Q_ce_loss_change": Q, "Q_raw_ce_loss_change": _f(q["loss_change"]), "isolated_accuracy_change_pp": _f(q["accuracy_change_pp"]),
                             "per_flip_drop_pp": pf, "R_exposure_unprotected_pp": R_none, "residual_rate_single_bit": f_w, "R_exposure_pp": R_none * f_w,
                             "R_scale_unprotected_pp": Rs_none, "residual_rate_scale_word": f_s, "R_scale_pp": Rs_none * f_s,
                             "S_total_bytes": S, **{f"S_{k}": v for k, v in parts_b.items()}})
    meta = {"layers": len(layers), "options_per_layer": len(all_options()), "candidates": len(rows), "quantized_weights_N": N,
            "definitions": __doc__, "per_flip_group_detail_example": parts}
    return rows, meta


def rows_to_problem(rows: Sequence[dict], mode: str = "joint", q_col: str = "Q_ce_loss_change", r_col: str = "R_exposure_pp"):
    layers = list(OrderedDict.fromkeys(r["layer_name"] for r in rows))
    options = [o for o in all_options() if any(r["option"] == o for r in rows)]
    if mode == "datatype_only":
        options = [o for o in options if "+" not in o]
    elif mode != "joint":
        raise ValueError(mode)
    L, D = len(layers), len(options)
    arr = {k: np.zeros((L, D)) for k in ("Q", "R", "S", "Rs", "n")}
    seen = set()
    for r in rows:
        if r["option"] not in options:
            continue
        i, j = layers.index(r["layer_name"]), options.index(r["option"])
        if (i, j) in seen:
            raise ValueError("duplicate candidate row")
        seen.add((i, j))
        arr["Q"][i, j], arr["R"][i, j], arr["S"][i, j] = _f(r[q_col]), _f(r[r_col]), _f(r["S_total_bytes"])
        arr["Rs"][i, j], arr["n"][i, j] = _f(r["R_scale_pp"]), _f(r["num_weights"])
    if len(seen) != L * D:
        raise ValueError("candidate table is not a full layer x option grid")
    return S5.Problem(layers, options, arr["Q"], arr["R"], arr["S"], Rscale=arr["Rs"], dce=arr["Q"]), arr["n"][:, 0]


def objective_of(p: S5.Problem, choice: Sequence[int], lam: float, norm: str = "minmax") -> float:
    qn = S5.normalize(p.Q.reshape(-1), norm).reshape(p.Q.shape)
    rn = S5.normalize(p.R.reshape(-1), norm).reshape(p.Q.shape)
    idx = np.arange(len(p.layers))
    return float((qn + lam * rn)[idx, list(choice)].sum())


def run_sweep(problems: Dict[str, Tuple[S5.Problem, np.ndarray]], lambdas, avg_bits, taus, norms, tau_scale: float) -> List[dict]:
    """Every (mode, normalisation, lambda, B, tau_fault) setting solved exactly (HiGHS). tau None = no limit."""
    N = float(next(iter(problems.values()))[1].sum())
    rows = []
    for mode, (p, _) in problems.items():
        for norm in norms:
            for lam in lambdas:
                for bits in avg_bits:
                    for tau in [None] + list(taus):
                        B = bits * N / 8.0
                        ts = tau_scale if mode == "joint" else None
                        sol = S5.solve(p, lam, norm, "weights", max_bytes=B, max_sum_R=tau, tau_scale=ts)
                        row = {"mode": mode, "normalization": norm, "lambda": lam, "B_avg_bits_per_weight": bits, "B_bytes": B, "tau_fault_pred_drop_pp": "" if tau is None else tau,
                               "tau_scale_pp": "" if ts is None else ts, "status": sol["status"], "objective": "", "storage_bytes": "", "avg_bits_per_weight": "", "sum_Q": "",
                               "pred_weight_fault_drop_pp": "", "pred_max_scale_word_risk_pp": "", "n_protected_layers": "", "assignment": ""}
                        if sol["choice"] is not None:
                            t = p.totals(sol["choice"])
                            assert t["storage_bytes"] <= B + 1e-6
                            row.update({"objective": sol["objective"], "storage_bytes": int(t["storage_bytes"]), "avg_bits_per_weight": 8 * t["storage_bytes"] / N,
                                        "sum_Q": t["sum_Q"], "pred_weight_fault_drop_pp": t["sum_R_pp"], "pred_max_scale_word_risk_pp": t["max_layer_scale_risk_pp"],
                                        "n_protected_layers": sum(1 for c in sol["choice"] if "+" in p.formats[c]),
                                        "assignment": ";".join(short_token(p.formats[c]) for c in sol["choice"])})
                        rows.append(row)
    return rows


def distinct_assignments(sweep_rows: Sequence[dict]) -> List[dict]:
    cnt, modes = Counter(), {}
    for r in sweep_rows:
        if r["status"] == "optimal":
            cnt[r["assignment"]] += 1
            modes.setdefault(r["assignment"], set()).add(r["mode"])
    return [{"assignment_id": f"A{i:03d}", "found_in_modes": "|".join(sorted(modes[a])), "n_settings": c, "assignment": a} for i, (a, c) in enumerate(cnt.most_common())]


# ------------------------------------------------------------------------------------------------ measured end-to-end selection (pure functions)
def feasible_items(items: Sequence[dict], delta: float, tau: float, bits: float, tau_s: Optional[float]) -> List[dict]:
    """Assignments whose MEASURED metrics satisfy the constraints (clean drop vs FP32, weights_k8 mean drop, average bits, scales_k1 mean drop)."""
    def r_scale(it):
        s = it["fault_end_to_end"].get("scales_k1")
        return float(s["drop_pp_mean"]) if s else 0.0
    return [it for it in items if "weights_k8" in it["fault_end_to_end"] and it["clean_validation"]["drop_pp_vs_fp32"] <= delta + 1e-9
            and float(it["fault_end_to_end"]["weights_k8"]["drop_pp_mean"]) <= tau + 1e-9 and it["storage"]["avg_bits_per_weight"] <= bits + 1e-9
            and (tau_s is None or r_scale(it) <= tau_s + 1e-9)]


def best_item(items, delta, tau, bits, tau_s):
    f = feasible_items(items, delta, tau, bits, tau_s)
    return min(f, key=lambda it: (it["objective_value"], it["storage"]["actual_stored_bytes"], it["assignment_id"])) if f else None


# ------------------------------------------------------------------------------------------------ protected faults on an assembled model
def _u8(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr).view(np.uint8)


def read_word(words: np.ndarray, w: int) -> int:
    """32-bit little-endian word ``w`` of the serialized payload of ``words`` (zero padded past the end)."""
    b = _u8(words)[4 * w:4 * w + 4]
    return int.from_bytes(b.tobytes() + b"\x00" * (4 - b.size), "little")


def n_payload_words(words: np.ndarray) -> int:
    return -(-int(words.nbytes) // 4)


class ProtectedFaults:
    """Maps faults on the STORED bits of a protected assembled model to the residual faults that survive each layer's decoder.

    ``prots[l]`` is the layer's protection token part (none / parity / secded / dmr / scale_tmr). Faults are given on stored bits; the residual data-bit flips
    (what the decoded weights actually see) are obtained by encoding the real stored word, flipping the stored bits and reading the word back through the
    real decoder (``fr.ecc.outcome``). The redundancy bits are simulated by the decoder logic, they are not separate arrays.
    """

    def __init__(self, store, prots: Sequence[str]):
        self.store, self.prots = store, list(prots)
        self.width = [int(w) for w in store.width]
        self.nbits = [int(n) * w for n, w in zip(store.n, self.width)]            # real payload bits of each layer
        self.nwords = [n_payload_words(w) for w in store.words]
        self._cand = None

    # ---- weight data bits (declared distribution): flips are (layer, element, bit-of-element)
    def residual_from_data_flips(self, flips: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        by = {}
        for l, e, b in flips:
            g = e * self.width[l] + b
            by.setdefault((l, g // 32), []).append(g % 32)
        out = []
        for (l, w), bits in by.items():
            out += self._word_residual(l, w, [(ecc.stored_position(self._wp(l), gb)) for gb in bits])
        return sorted(out)

    def _wp(self, l: int) -> str:
        return "none" if self.prots[l] == TMR else self.prots[l]

    def _word_residual(self, l: int, w: int, stored_positions: Sequence[int]) -> List[Tuple[int, int, int]]:
        wp = self._wp(l)
        data = read_word(self.store.words[l], w)
        o = ecc.outcome(wp, np.array([data], dtype=np.uint64), np.array([list(stored_positions)], dtype=np.int64))
        diff = int(o["data_out"][0]) ^ data
        res = []
        for j in range(32):
            if (diff >> j) & 1:
                g = w * 32 + j
                if g < self.nbits[l]:                                              # padding bits are never decoded
                    res.append((l, g // self.width[l], g % self.width[l]))
        return res

    # ---- stored-bit distribution (data bits + redundancy bits of the protected words)
    def _candidates(self):
        """per layer: (words, candidates per full word, candidates of the last word)."""
        if self._cand is None:
            out = []
            for l in range(len(self.prots)):
                wp = self._wp(l)
                nw = self.nwords[l]
                last_valid = self.nbits[l] - 32 * (nw - 1)
                out.append((nw, self._cands_of(wp, 32), self._cands_of(wp, last_valid), last_valid))
            self._cand = out
        return self._cand

    @staticmethod
    def _cands_of(wp: str, valid: int) -> int:
        return {"none": valid, "parity": valid + 1, "secded": valid + 7, "dmr": 2 * valid}[wp]

    @staticmethod
    def stored_positions_of(wp: str, valid: int) -> List[int]:
        """Stored-bit positions of the candidates of one word, data bits first."""
        data = [ecc.stored_position(wp, b) for b in range(valid)]
        if wp == "dmr":
            return data + [32 + b for b in range(valid)]
        return data + ecc.check_positions(wp)

    def draw_stored_flips(self, rng: np.random.Generator, k: int) -> List[Tuple[int, int, int]]:
        """k DISTINCT stored bits uniform over all candidate stored bits of all payload words -> [(layer, word, stored position)]."""
        cand = self._candidates()
        per_layer = [(nw - 1) * cf + cl for nw, cf, cl, _ in cand]
        total = int(sum(per_layer))
        pick = rng.choice(total, size=k, replace=False)
        cum = np.cumsum(per_layer)
        out = []
        for x in pick:
            l = int(np.searchsorted(cum, x, side="right"))
            idx = int(x - (cum[l] - per_layer[l]))
            nw, cf, cl, lv = cand[l]
            if idx < (nw - 1) * cf:
                w, off, valid = idx // cf, idx % cf, 32
            else:
                w, off, valid = nw - 1, idx - (nw - 1) * cf, lv
            out.append((l, int(w), self.stored_positions_of(self._wp(l), valid)[off]))
        return out

    def residual_from_stored_flips(self, flips: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        by = {}
        for l, w, pos in flips:
            by.setdefault((l, w), []).append(pos)
        out = []
        for (l, w), poss in by.items():
            out += self._word_residual(l, w, poss)
        return sorted(out)

    # ---- scale words
    def scale_positions(self) -> List[Tuple[int, int, int]]:
        """Every stored scale bit as (layer, copy / 0, stored position): 32 per unprotected word, 3 x 32 scale_tmr, 32 + redundancy for word-level protections."""
        out = []
        for l, c in enumerate(self.store.scale_copies):
            if c is None:
                continue
            if self.prots[l] == TMR:
                out += [(l, k, b) for k in range(3) for b in range(32)]
            else:
                wp = self._wp(l)
                out += [(l, 0, p) for p in self.stored_positions_of(wp, 32)]
        return out

    def residual_from_scale_flips(self, flips: Sequence[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        """(layer, copy, stored position) -> residual scale flips (layer, copy, data bit) after the layer's decoder."""
        by, out = {}, []
        for l, c, pos in flips:
            if self.prots[l] == TMR:
                out.append((l, c, pos))                                              # the vote happens inside the store
            else:
                by.setdefault(l, []).append(pos)
        for l, poss in by.items():
            pattern = int(self.store.scale_copies[l][0])
            o = ecc.outcome(self._wp(l), np.array([pattern], dtype=np.uint64), np.array([poss], dtype=np.int64))
            diff = int(o["data_out"][0]) ^ pattern
            out += [(l, 0, j) for j in range(32) if (diff >> j) & 1]
        return sorted(out)


def stored_bytes(store, prots: Sequence[str]) -> dict:
    """Actual stored bytes of the assembled model (payload + scale + alignment + packed redundancy bits)."""
    tot = {"payload_bytes": 0, "scale_bytes": 0, "alignment_padding_bytes": 0, "redundancy_bytes": 0}
    for l, p in enumerate(prots):
        payload = int(store.words[l].nbytes)
        scale = 4 if store.scale_copies[l] is not None else 0
        tot["payload_bytes"] += payload
        tot["scale_bytes"] += scale
        if p == TMR:
            tot["redundancy_bytes"] += P.TMR_OVERHEAD_BYTES
        else:
            sb = ecc.storage_bytes(p, payload, scale)
            tot["alignment_padding_bytes"] += sb["alignment_padding_bytes"]
            tot["redundancy_bytes"] += sb["redundancy_bytes"]
    tot["total_bytes"] = sum(tot.values())
    return tot


def draw_trial_faults(pf: ProtectedFaults, dist: str, seed: int, k: int):
    """-> (stored faults, residual weight flips, residual scale flips) for one trial of a declared distribution."""
    from . import stage5_e2e as E2
    rng = np.random.default_rng(seed)
    if dist == "weights_k8":
        orig = E2.draw_global_weight_flips(rng, pf.store, k)
        return {"weights": [list(f) for f in orig]}, pf.residual_from_data_flips(orig), []
    if dist == "stored_k8":
        orig = pf.draw_stored_flips(rng, k)
        return {"stored_words": [list(f) for f in orig]}, pf.residual_from_stored_flips(orig), []
    if dist == "scales_k1":
        pos = pf.scale_positions()
        if not pos:
            raise ValueError("no INT layer: no scale words")
        pick = rng.choice(len(pos), size=k, replace=False)
        orig = [pos[int(i)] for i in pick]
        return {"scales": [list(f) for f in orig]}, [], pf.residual_from_scale_flips(orig)
    raise ValueError(dist)


def run_protected_trial(inj, pf: ProtectedFaults, X, y, clean: dict, ctx: dict, *, assignment_id: str, dist: str, trial: int, base_seed: int) -> dict:
    """One end-to-end fault trial on the assembled model: draw faults on the stored bits, pass them through every layer's decoder, apply the residual flips to the
    stored data bytes, evaluate, restore byte-exactly. The record keeps both the stored faults and the residual ones."""
    from . import stage5_e2e as E2
    k = DIST_K[dist]
    seed = S3.trial_seed(base_seed, f"mixed:{assignment_id}", "GLOBAL", dist, "all", k, trial)
    stored, rw, rs = draw_trial_faults(pf, dist, seed, k)
    rec = E2.run_mixed_trial(inj, X, y, clean, ctx, assignment_id=assignment_id, dist=dist, k=k, trial=trial, base_seed=base_seed,
                             explicit_weight_flips=rw, explicit_scale_flips=rs)
    rec.update({"requested_k": k, "stored_faults": stored, "residual_fault_count": len(rw) + len(rs), "schema": SCHEMA, "seed": seed,
                "protection_per_layer_hash": hashlib.sha256(",".join(pf.prots).encode()).hexdigest()})
    return rec


def uniform_assignments(layers: int) -> "OrderedDict[str, List[str]]":
    out = OrderedDict()
    for f in FORMATS:
        out[f"U_{SHORT[f]}"] = [f] * layers
    for f in ("int8_pt", "int4_pt"):
        out[f"U_{SHORT[f]}_TMR"] = [option_token(f, TMR)] * layers
    for f in ("int8_pt", "fp8_e4m3fn"):
        out[f"U_{SHORT[f]}_SECDED"] = [option_token(f, "secded")] * layers
    return out
