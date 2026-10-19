"""F6 length-correct diagnostic - exploratory, separate from F6 (D58).

Implements ``docs/F6_LENGTH_CORRECT_DIAGNOSTIC.md`` (approved protocol,
SHA-256 097cbf8a...6eb1). F6's registered result, threshold,
``leak_suspected = false``, interpretation and ``permute_labels`` are not
touched: this script only reads ``runs/main`` and writes to
``runs/main_f6_diag/``. It gives no leakage verdict. CPU only; no torch import.

Three phases, each strictly larger than the last::

    python scripts/diag_f6_length_correct.py --phase preflight  # checks only; no fit or write
    python scripts/diag_f6_length_correct.py --phase gate       # + seed-12 reproduction; no write
    python scripts/diag_f6_length_correct.py --phase run        # + diagnostic -> runs/main_f6_diag/

Every phase refuses to start if ``runs/main_f6_diag/`` exists, verifies the
registered ``robustness.json`` SHA-256 before and after, and checks that the
registered functions it reuses are byte-for-byte the approved ones. ``run``
creates the output directory only after the gate has passed; a gate failure,
a runtime-budget stop or a failed preflight writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from _stage import ROOT

from nlaast import config as config_mod
from nlaast.baselines import probe as probe_mod
from nlaast.logging_utils import read_jsonl, write_json

# --- fixed by the protocol -------------------------------------------------

RUN_ID = "main_f6_diag"
SOURCE_RUN = "main"
PROTOCOL_SHA256 = "097cbf8a60b381b17b7973c50684153c1af85eea55b838b43b264504baf66eb1"
ROBUSTNESS_SHA256 = "45701a51c98d5ebd936ed1fc6a463f7fb01f1e2743378094ee7bd6cac640d5ed"

GATE_TOLERANCE = 1e-9
F6_SHUFFLE_SEED = 12                   # probe.seed + 1, what cross_validate passes
SEED_ROOT, SEED_TAG = 11, 0xF6
STREAM_FULL, STREAM_SUBSET, STREAM_STRATA = 0, 1, 2

B_DEFAULT, B_FALLBACK = 100, 50
FITS_PER_DRAW = 5                      # S0, S1 full; S0, S1, S2 subset
RUNTIME_OVERHEAD = 1.25
B_DEFAULT_LIMIT_S = 6 * 3600
B_FALLBACK_LIMIT_S = 12 * 3600

N_BOOTSTRAP = 2000
PERCENTILES = (2.5, 97.5)
DRAW_SIGN_SHARE = 0.975                # "excludes 0" for the donor-draw summary

LABEL = ("EXPLORATORY DIAGNOSTIC (docs/F6_LENGTH_CORRECT_DIAGNOSTIC.md, D58). "
         "Separate from F6: F6's registered result, threshold, leak_suspected = false "
         "and interpretation are unchanged. Descriptive only; no leakage verdict.")

#: Written into summary.json beside the statistics (protocol "Uncertainty" and
#: "Positional oracle").
CAVEATS = {
    "donor_draw_percentiles": (
        "donor_draws p2_5 and p97_5 are the 2.5th-97.5th percentiles of individual "
        "draws: they describe how single draws vary, and are not an interval for the mean"),
    "monte_carlo_se": (
        "monte_carlo_se_of_mean (SD / sqrt(B)) is the precision of the mean over donor "
        "draws only; it is not problem-level sampling or refitting uncertainty"),
    "separate_summaries": (
        "donor_draws and problem_bootstrap are separate summaries and are not a joint "
        "confidence statement; neither includes model-refitting variability, which is "
        "unmeasured"),
    "oracle": (
        "the positional oracle is a position-derived reference benchmark; it is not a "
        "guaranteed upper bound, a formal null distribution or a definitive leakage test"),
    "probe_above_oracle": (
        "probe AUC above the oracle is a discrepancy to investigate, not proof of leakage"),
    "probe_at_or_below_oracle": (
        "probe AUC similar to or below the oracle does not establish that position fully "
        "explains the predictability, and does not rule out leakage"),
    "verdict": "this output gives no leakage verdict; F6's registered verdict is unchanged",
}


@dataclass(frozen=True)
class Registered:
    """What the gate and the data checks must reproduce before any new fit."""

    n_labels_changed: int = 718
    shuffled_auc: float = 0.6142873504
    real_auc: float = 0.9460944231
    n_rows: int = 2681
    n_problems: int = 250
    n_eligible_problems: int = 246
    n_eligible_rows: int = 2577


#: SHA-256 of ``inspect.getsource`` of each registered function this script
#: reuses, taken at commit 0f82bd2 (unchanged since 23d0e3b). A mismatch means
#: the registered code changed and the diagnostic no longer measures what the
#: protocol describes.
SOURCE_FINGERPRINTS = {
    "nlaast.baselines.probe.permute_labels":
        "f5d9b0274ccc1eced125dcad8810af9c4f0e1bc431d927f6de1e6669cd80b62d",
    "nlaast.baselines.probe.cross_validate":
        "a9ec8dbc4e6cb84f8c05956520ebaa3385ace7fc0e9eb64b9cf7dc183eba3214",
    "nlaast.baselines.probe.grouped_folds":
        "adbe26e59aae230184fc03cb506f2cbfdd0d6adb6461387ae3f592919416d31a",
    "nlaast.baselines.probe.CorrectnessProbe":
        "6b106469816d6b85bf9d86aa1149defecf5d416b7f632b33abe2cfd0521f52d9",
    "nlaast.baselines.probe._metrics":
        "0ff4967823fe5a31951cbf95634b4a938e17606846b57e7455fcd8fd04b15590",
    "run_falsification_tests.f6_probe_leakage":
        "e9a44c6beee1f76dad5aa69ae56029afcea5e623135d3f53dfde7b35ab729492",
    "run_falsification_tests.chunks_of":
        "55ae0a8d28f5e15a77a4659c5cf23afdd1412fda152cfb3761c2b8d6ba8a212e",
    "rq2_claim_probe.boundary_position":
        "f56bc5b0aa7c8ee5f5476db5bf741b37f3cf6a904719c448b20b1fb3d5f7845a",
}


class DiagnosticStop(Exception):
    """A stop the protocol requires. ``code`` is the process exit status."""

    code = 1

    def __init__(self, reason: str, details: Any = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details


class PreflightFailure(DiagnosticStop):
    code = 2


class GateFailure(DiagnosticStop):
    code = 3


class BudgetStop(DiagnosticStop):
    code = 4


class RunFailure(DiagnosticStop):
    code = 1


# --- hashing ----------------------------------------------------------------

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_text_lf(path: Path) -> str:
    """SHA-256 of a text file with CRLF folded to LF - the committed blob's
    content, whatever ``core.autocrlf`` did to the working copy."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_fingerprints() -> dict[str, str]:
    import rq2_claim_probe
    import run_falsification_tests

    objs = {
        "nlaast.baselines.probe.permute_labels": probe_mod.permute_labels,
        "nlaast.baselines.probe.cross_validate": probe_mod.cross_validate,
        "nlaast.baselines.probe.grouped_folds": probe_mod.grouped_folds,
        "nlaast.baselines.probe.CorrectnessProbe": probe_mod.CorrectnessProbe,
        "nlaast.baselines.probe._metrics": probe_mod._metrics,
        "run_falsification_tests.f6_probe_leakage": run_falsification_tests.f6_probe_leakage,
        "run_falsification_tests.chunks_of": run_falsification_tests.chunks_of,
        "rq2_claim_probe.boundary_position": rq2_claim_probe.boundary_position,
    }
    return {k: hashlib.sha256(inspect.getsource(v).encode("utf-8")).hexdigest()
            for k, v in objs.items()}


# --- length-fitting rules ----------------------------------------------------

def phi(n_d: int, n_r: int) -> np.ndarray:
    """S1's donor index for each receiver row: floor(j (n_d-1)/(n_r-1) + 1/2).

    Integer arithmetic, so there is no floating-point rounding at the halves.
    Undefined for a one-row receiver (the denominator is n_r - 1); F6's rows
    per problem are 4 to 34, so it never arises on ``main``.
    """
    if n_d < 1 or n_r < 1:
        raise ValueError(f"empty sequence (n_d={n_d}, n_r={n_r})")
    if n_r == 1:
        raise ValueError("phi is undefined for a one-row receiver (n_r - 1 = 0)")
    j = np.arange(n_r, dtype=np.int64)
    return (2 * j * (n_d - 1) + (n_r - 1)) // (2 * (n_r - 1))


def resize_indices(n_d: int, n_r: int) -> np.ndarray:
    """The donor index ``np.resize`` takes for each receiver row."""
    if n_d < 1:
        raise ValueError("empty donor")
    return np.arange(n_r, dtype=np.int64) % n_d


def verbatim_indices(n_d: int, n_r: int) -> np.ndarray:
    """S2: the donor sequence copied as it is; lengths must match."""
    if n_d != n_r:
        raise ValueError(f"S2 donor length {n_d} != receiver length {n_r}")
    return np.arange(n_r, dtype=np.int64)


INDEXERS: dict[str, Callable[[int, int], np.ndarray]] = {
    "S0": resize_indices, "S1": phi, "S2": verbatim_indices,
}


def fit_to_length(rule: str, y_d: np.ndarray, n_r: int) -> np.ndarray:
    y_d = np.asarray(y_d, dtype=int)
    if rule == "S0":
        return np.resize(y_d, n_r)                 # exactly what permute_labels does
    return y_d[INDEXERS[rule](len(y_d), n_r)]


# --- donor assignments -------------------------------------------------------

def stream(stream_id: int, b: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED_ROOT, SEED_TAG, stream_id, b]))


def f6_permutation(n: int, seed: int = F6_SHUFFLE_SEED) -> np.ndarray:
    """F6's own draw, step for step as ``permute_labels`` makes it (draw 0)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n > 1:
        for _ in range(10):
            if all(perm[i] != i for i in range(n)):
                break
            perm = rng.permutation(n)
    return perm


def rejection_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform permutation redrawn until no element is fixed."""
    if n < 2:
        raise ValueError(f"no derangement of {n} element(s)")
    ident = np.arange(n)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == ident):
            return perm


def donors_from_permutation(order: Sequence[str], perm: np.ndarray) -> dict[str, str]:
    """Receiver ``order[i]`` takes the labels of ``order[perm[i]]`` - the
    convention of ``permute_labels`` (groups in sorted order)."""
    return {order[i]: order[int(p)] for i, p in enumerate(perm)}


def full_donors(order: Sequence[str], b: int) -> dict[str, str]:
    perm = f6_permutation(len(order)) if b == 0 else \
        rejection_derangement(len(order), stream(STREAM_FULL, b))
    return donors_from_permutation(order, perm)


def subset_donors(order: Sequence[str], b: int) -> dict[str, str]:
    return donors_from_permutation(order, rejection_derangement(len(order),
                                                                stream(STREAM_SUBSET, b)))


def stratum_donors(lengths: dict[str, int], b: int) -> dict[str, str]:
    """S2: a derangement inside each length stratum, strata in ascending
    length, one generator per draw consumed stratum by stratum."""
    rng = stream(STREAM_STRATA, b)
    out: dict[str, str] = {}
    for n in sorted(set(lengths.values())):
        members = sorted(p for p, m in lengths.items() if m == n)
        out.update(donors_from_permutation(members, rejection_derangement(len(members), rng)))
    return out


def exact_length_eligible(lengths: dict[str, int]) -> tuple[list[str], list[str]]:
    """Problems whose row count another problem shares (eligible), and the rest."""
    count: dict[int, int] = {}
    for n in lengths.values():
        count[n] = count.get(n, 0) + 1
    eligible = sorted(p for p, n in lengths.items() if count[n] >= 2)
    excluded = sorted(p for p, n in lengths.items() if count[n] < 2)
    return eligible, excluded


# --- inputs ----------------------------------------------------------------

@dataclass
class F6Inputs:
    """F6's rows, in F6's order, with the positional quantities the protocol uses.

    ``row_index`` is j, the position of the row among its problem's F6 rows
    (S1's and the oracle's position axis); ``rel`` is chunk-relative position
    (the audit's label-position axis); ``P`` holds A1's four position features.
    """

    X: np.ndarray
    y: np.ndarray
    groups: list[str]
    row_index: np.ndarray
    rel: np.ndarray
    P: np.ndarray
    order: list[str] = field(init=False)
    rows: dict[str, np.ndarray] = field(init=False)
    labels: dict[str, np.ndarray] = field(init=False)
    lengths: dict[str, int] = field(init=False)
    problem_of_row: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.y = np.asarray(self.y, dtype=int)
        self.order = sorted(set(self.groups))
        rows: dict[str, list[int]] = {}
        for i, g in enumerate(self.groups):
            rows.setdefault(g, []).append(i)
        self.rows = {g: np.asarray(v, dtype=np.int64) for g, v in rows.items()}
        self.labels = {g: self.y[v] for g, v in self.rows.items()}
        self.lengths = {g: len(v) for g, v in self.rows.items()}
        for g, idx in self.rows.items():
            if not np.array_equal(self.row_index[idx], np.arange(len(idx))):
                raise ValueError(f"row_index of {g} is not 0..n-1 in row order")
        pos = {g: k for k, g in enumerate(self.order)}
        self.problem_of_row = np.array([pos[g] for g in self.groups], dtype=np.int64)

    def subset(self, problems: Sequence[str]) -> "F6Inputs":
        keep = set(problems)
        idx = np.array([i for i, g in enumerate(self.groups) if g in keep], dtype=np.int64)
        return F6Inputs(X=self.X[idx], y=self.y[idx], groups=[self.groups[i] for i in idx],
                        row_index=self.row_index[idx], rel=self.rel[idx], P=self.P[idx])


def load_f6_inputs(cfg) -> tuple[F6Inputs, dict]:
    """F6's assembly, as ``f6_probe_leakage`` builds it (same traces, same
    filter, same loop), plus the positions. The gate verifies the result."""
    from nlaast.activations.store import ActivationStore
    from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
    from nlaast.trace.chunking import prefix_text
    from rq2_claim_probe import boundary_position, position_matrix
    from run_falsification_tests import chunks_of

    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer
    X, y, groups, ci_list, rel, positions = [], [], [], [], [], []
    for pid, t in traces.items():
        if not store.has(pid, layer):
            continue
        arr, meta = store.get(pid, layer)
        chunks = chunks_of(t)
        for row_i, ci in enumerate(meta.get("chunk_indices", [])):
            if ci >= len(chunks):
                continue
            ans = extract_answer(prefix_text(chunks, ci, t["trace"]), PERMISSIVE)
            if ans is None:
                continue
            X.append(arr[row_i])
            y.append(1 if equivalent(ans, t.get("gold")) else 0)
            groups.append(pid)
            ci_list.append(int(ci))
            n = len(chunks)
            rel.append(ci / (n - 1) if n > 1 else 1.0)
            positions.append(boundary_position(t, ci))
    row_index, seen = [], {}
    for g in groups:
        row_index.append(seen.get(g, 0))
        seen[g] = seen.get(g, 0) + 1
    ascending = all(
        all(a < b for a, b in zip(c, c[1:]))
        for c in ({g: [ci for ci, h in zip(ci_list, groups) if h == g] for g in seen}).values())
    inputs = F6Inputs(X=np.vstack(X).astype(np.float64), y=np.asarray(y), groups=groups,
                      row_index=np.asarray(row_index), rel=np.asarray(rel, dtype=float),
                      P=position_matrix(positions))
    return inputs, {"chunk_indices_ascending_within_problem": bool(ascending)}


# --- shuffled labels, oracle, label statistics ----------------------------------

def shuffled_labels(inp: F6Inputs, donors: dict[str, str], rule: str) -> np.ndarray:
    out = np.empty_like(inp.y)
    for g, idx in inp.rows.items():
        out[idx] = fit_to_length(rule, inp.labels[donors[g]], len(idx))
    return out


def oracle_scores(inp: F6Inputs, donors: dict[str, str], rule: str) -> np.ndarray:
    """g_r(j): the mean, over every problem except r and its donor, of the
    label rule ``rule`` gives row j of r from that problem's real labels."""
    n_problems = len(inp.order)
    if n_problems < 3:
        raise ValueError("the oracle needs at least three problems")
    totals: dict[int, np.ndarray] = {}
    g = np.empty(len(inp.y), dtype=float)
    for r, idx in inp.rows.items():
        if donors[r] == r:
            # D_r = all problems except r and pi_b(r) presumes a derangement;
            # a fixed point would subtract r twice.
            raise ValueError(f"oracle receiver {r} is its own donor; the oracle needs a "
                             "derangement")
        n_r = len(idx)
        if n_r not in totals:
            totals[n_r] = np.sum([fit_to_length(rule, inp.labels[d], n_r) for d in inp.order],
                                 axis=0).astype(float)
        own = fit_to_length(rule, inp.labels[r], n_r)
        don = fit_to_length(rule, inp.labels[donors[r]], n_r)
        g[idx] = (totals[n_r] - own - don) / (n_problems - 2)
    return g


def label_statistics(inp: F6Inputs, y_new: np.ndarray, donors: dict[str, str] | None,
                     rule: str | None) -> dict:
    """Stage 0: label statistics of one labelling (no model)."""
    per_problem = {g: float(np.mean(y_new[idx])) for g, idx in inp.rows.items()}
    transitions, monotone = 0, 0
    for g, idx in inp.rows.items():
        d = np.diff(y_new[idx])
        transitions += int(np.count_nonzero(d))
        monotone += int(np.all(d >= 0))
    out: dict[str, Any] = {
        "prevalence": float(np.mean(y_new)),
        "per_problem_prevalence": [per_problem[g] for g in inp.order],
        "n_transitions": transitions,
        "monotone_problem_share": monotone / len(inp.order),
        "label_relative_position_auc": auc(y_new, inp.rel),
    }
    if donors is not None and rule is not None:
        dup = drop = 0
        shift = []
        for g, idx in inp.rows.items():
            n_r, n_d = len(idx), inp.lengths[donors[g]]
            used = len(np.unique(INDEXERS[rule](n_d, n_r)))
            dup += n_r - used
            drop += n_d - used
            shift.append(abs(per_problem[g] - float(np.mean(inp.labels[donors[g]]))))
        out.update({
            "n_labels_changed": int(np.sum(y_new != inp.y)),
            "n_labels_duplicated": int(dup),
            "n_labels_dropped": int(drop),
            "per_problem_prevalence_shift_vs_donor": {"mean": float(np.mean(shift)),
                                                      "max": float(np.max(shift))},
            "fixed_points": sum(1 for g in inp.order if donors[g] == g),
        })
    return out


# --- AUC and uncertainty ------------------------------------------------------

def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Pooled AUC as F6 computes it (``roc_auc_score``); nan for one class."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=float)))


def weighted_auc(y: np.ndarray, score: np.ndarray, problem: np.ndarray,
                 counts: np.ndarray) -> np.ndarray:
    """AUC for each bootstrap replicate, a row of ``counts`` giving how many
    times each problem was drawn. Ties count one half, as in ``roc_auc_score``;
    with all counts 1 it equals :func:`auc`. nan where a replicate has one class."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    grp = np.concatenate([[0], np.cumsum(s[1:] != s[:-1])]) if len(s) else np.array([], int)
    n_groups = int(grp[-1]) + 1 if len(s) else 0
    pr, ys = problem[order], y[order]
    a_pos = np.zeros((counts.shape[1], n_groups))
    a_neg = np.zeros((counts.shape[1], n_groups))
    np.add.at(a_pos, (pr[ys == 1], grp[ys == 1]), 1.0)
    np.add.at(a_neg, (pr[ys == 0], grp[ys == 0]), 1.0)
    out = np.empty(counts.shape[0])
    for lo in range(0, counts.shape[0], 250):
        c = counts[lo:lo + 250].astype(float)
        pos, neg = c @ a_pos, c @ a_neg
        below = np.cumsum(neg, axis=1) - neg
        num = (pos * (below + 0.5 * neg)).sum(axis=1)
        p_tot, n_tot = pos.sum(axis=1), neg.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[lo:lo + 250] = np.where((p_tot > 0) & (n_tot > 0), num / (p_tot * n_tot), np.nan)
    return out


def bootstrap_counts(n_problems: int, n_boot: int, seed: int) -> np.ndarray:
    """Problem multiplicities for each resample (with replacement)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_problems, size=(n_boot, n_problems))
    return np.stack([np.bincount(d, minlength=n_problems) for d in draws])


def monte_carlo_se(values: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def draw_summary(values: Sequence[float]) -> dict:
    """Donor-draw variability: the spread of single-draw values."""
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    if len(ok) == 0:
        return {"n_draws": int(len(v)), "n_undefined": int(len(v)), "mean": None}
    return {
        "n_draws": int(len(v)), "n_undefined": int(len(v) - len(ok)),
        "mean": float(np.mean(ok)),
        "sd": float(np.std(ok, ddof=1)) if len(ok) > 1 else None,
        "p2_5": float(np.percentile(ok, PERCENTILES[0])),
        "p97_5": float(np.percentile(ok, PERCENTILES[1])),
        "share_positive": float(np.mean(ok > 0)), "share_negative": float(np.mean(ok < 0)),
        "monte_carlo_se_of_mean": monte_carlo_se(ok),
    }


def bootstrap_summary(replicates: np.ndarray) -> dict:
    """Problem-level sampling variability of the draw-averaged statistic."""
    r = np.asarray(replicates, dtype=float)
    ok = r[np.isfinite(r)]
    if len(ok) == 0:
        return {"n_resamples": int(len(r)), "n_undefined": int(len(r)), "mean": None}
    return {"n_resamples": int(len(r)), "n_undefined": int(len(r) - len(ok)),
            "mean": float(np.mean(ok)),
            "p2_5": float(np.percentile(ok, PERCENTILES[0])),
            "p97_5": float(np.percentile(ok, PERCENTILES[1]))}


WORDING = {
    "both": ("consistent in sign across donor draws, and the draw-averaged value is stable "
             "under problem resampling (two separate summaries, not a joint confidence "
             "statement)"),
    "draws_only": ("consistent across donor draws for this dataset, but not "
                   "distinguishable from problem-sampling variability"),
    "bootstrap_only": ("the draw-averaged value is stable under problem resampling, but "
                       "individual draws vary in sign; a single draw would not reliably "
                       "show it"),
    "neither": "no consistent difference in either summary",
    "conflicting": "reported as conflicting; no direction stated",
}


def wording(draws: dict, boot: dict) -> dict:
    """The protocol's predeclared wording for a difference statistic."""
    d_sign = 0
    if draws.get("mean") is not None and draws["n_undefined"] == 0:
        if draws["share_positive"] >= DRAW_SIGN_SHARE:
            d_sign = 1
        elif draws["share_negative"] >= DRAW_SIGN_SHARE:
            d_sign = -1
    b_sign = 0
    if boot.get("mean") is not None and boot["n_undefined"] == 0:
        if boot["p2_5"] > 0:
            b_sign = 1
        elif boot["p97_5"] < 0:
            b_sign = -1
    if d_sign and b_sign:
        key = "both" if d_sign == b_sign else "conflicting"
    elif d_sign:
        key = "draws_only"
    elif b_sign:
        key = "bootstrap_only"
    else:
        key = "neither"
    return {"case": key, "text": WORDING[key], "draw_sign": d_sign, "bootstrap_sign": b_sign,
            "complete": draws.get("n_undefined") == 0 and boot.get("n_undefined") == 0}


# --- checks -----------------------------------------------------------------

@dataclass
class Paths:
    root: Path
    source_dir: Path
    out_dir: Path
    robustness: Path
    protocol: Path
    decisions: Path
    script: Path

    @classmethod
    def default(cls, cfg) -> "Paths":
        return cls(root=ROOT, source_dir=cfg.dir, out_dir=cfg.dir.parent / RUN_ID,
                   robustness=cfg.dir / "robustness" / "robustness.json",
                   protocol=ROOT / "docs" / "F6_LENGTH_CORRECT_DIAGNOSTIC.md",
                   decisions=ROOT / "docs" / "DECISIONS.md",
                   script=Path(__file__).resolve())


def git_status(root: Path, paths: Sequence[Path]) -> dict:
    """Whether each path is tracked and unmodified, and the HEAD commit."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)

    out = {"head": git("rev-parse", "HEAD").stdout.strip(), "files": {}}
    for p in paths:
        rel = str(Path(p).resolve().relative_to(Path(root).resolve())).replace("\\", "/")
        tracked = git("ls-files", "--error-unmatch", rel).returncode == 0
        clean = git("status", "--porcelain", "--", rel).stdout.strip() == ""
        out["files"][rel] = {"tracked": tracked, "unmodified": clean}
    return out


def check_robustness(paths: Paths, when: str) -> str:
    h = sha256_file(paths.robustness)
    if h != ROBUSTNESS_SHA256:
        raise RunFailure(f"registered robustness.json SHA-256 changed ({when}): {h}")
    return h


def torch_loaded() -> bool:
    return "torch" in sys.modules


def preflight(paths: Paths, cfg_values: dict, git_fn: Callable = git_status,
              fingerprint_fn: Callable = source_fingerprints,
              torch_fn: Callable[[], bool] = torch_loaded) -> dict:
    """Every check that needs no data and no fit. Raises on any failure."""
    checks: dict[str, Any] = {}
    failures: list[str] = []

    def need(name: str, ok: bool, value: Any = None) -> None:
        checks[name] = {"ok": bool(ok), "value": value}
        if not ok:
            failures.append(name)

    need("output_dir_absent", not paths.out_dir.exists(), str(paths.out_dir))
    need("output_dir_is_the_diagnostic_run", paths.out_dir.name == RUN_ID
         and paths.out_dir.resolve() != paths.source_dir.resolve()
         and paths.source_dir.resolve() not in paths.out_dir.resolve().parents,
         str(paths.out_dir))
    h = sha256_file(paths.robustness) if paths.robustness.exists() else None
    need("registered_robustness_sha256", h == ROBUSTNESS_SHA256, h)
    p = sha256_file(paths.protocol) if paths.protocol.exists() else None
    need("approved_protocol_sha256", p == PROTOCOL_SHA256, p)
    fp = fingerprint_fn()
    need("registered_functions_unchanged", fp == SOURCE_FINGERPRINTS,
         {k: v == SOURCE_FINGERPRINTS.get(k) for k, v in fp.items()})
    script_sha = sha256_text_lf(paths.script)
    decisions = paths.decisions.read_text(encoding="utf-8") if paths.decisions.exists() else ""
    need("script_sha256_recorded_in_decisions", script_sha in decisions, script_sha)
    g = git_fn(paths.root, [paths.script, paths.protocol, paths.decisions])
    need("script_protocol_decisions_committed",
         all(f["tracked"] and f["unmodified"] for f in g["files"].values()), g)
    need("probe_config_as_protocol", cfg_values.get("probe") == {
        "kind": "logistic", "C": 1.0, "max_iter": 2000, "standardise": True, "n_folds": 5,
        "seed": 11}, cfg_values.get("probe"))
    need("torch_not_imported", not torch_fn())
    if failures:
        raise PreflightFailure(f"preflight failed: {', '.join(failures)}", checks)
    return checks


def data_checks(inp: F6Inputs, extra: dict, reg: Registered) -> dict:
    """Label-only checks on the assembled rows (no fit). Raises on failure."""
    eligible, excluded = exact_length_eligible(inp.lengths)
    donors0 = full_donors(inp.order, 0)
    y0 = shuffled_labels(inp, donors0, "S0")
    y_f6, _ = probe_mod.permute_labels(inp.y, inp.groups, "across_groups", F6_SHUFFLE_SEED)
    checks = {
        "n_rows": len(inp.y), "n_problems": len(inp.order),
        "n_eligible_problems": len(eligible),
        "n_eligible_rows": int(sum(inp.lengths[g] for g in eligible)),
        "excluded_problems": {g: inp.lengths[g] for g in excluded},
        "draw0_s0_equals_permute_labels": bool(np.array_equal(y0, y_f6)),
        "draw0_labels_changed": int(np.sum(y0 != inp.y)),
        "draw0_fixed_points": sum(1 for g in inp.order if donors0[g] == g),
        **extra,
    }
    expect = {"n_rows": reg.n_rows, "n_problems": reg.n_problems,
              "n_eligible_problems": reg.n_eligible_problems,
              "n_eligible_rows": reg.n_eligible_rows,
              "draw0_s0_equals_permute_labels": True,
              "draw0_labels_changed": reg.n_labels_changed,
              "draw0_fixed_points": 0,
              "chunk_indices_ascending_within_problem": True}
    bad = [k for k, v in expect.items() if k in checks and checks[k] != v]
    if bad:
        raise PreflightFailure(f"data checks failed: {', '.join(bad)}", checks)
    return checks


# --- gate and budget --------------------------------------------------------

def run_gate(inp: F6Inputs, probe_cfg, cv_fn: Callable, reg: Registered,
             clock: Callable[[], float] = time.perf_counter) -> dict:
    """The registered seed-12 shuffle and the real-label fit, reproduced."""
    t0 = clock()
    shuf = cv_fn(inp.X, inp.y, inp.groups, probe_cfg, shuffle_labels=True,
                 shuffle_scope="across_groups")
    t_s0 = clock() - t0
    real = cv_fn(inp.X, inp.y, inp.groups, probe_cfg)
    changed = (shuf.notes.get("shuffle") or {}).get("n_labels_changed")
    result = {
        "n_labels_changed": changed, "expected_n_labels_changed": reg.n_labels_changed,
        "shuffled_auc": shuf.auc, "expected_shuffled_auc": reg.shuffled_auc,
        "real_auc": real.auc, "expected_real_auc": reg.real_auc,
        "tolerance": GATE_TOLERANCE, "s0_fit_seconds": t_s0,
    }
    ok = (changed == reg.n_labels_changed
          and np.isfinite(shuf.auc) and abs(shuf.auc - reg.shuffled_auc) < GATE_TOLERANCE
          and np.isfinite(real.auc) and abs(real.auc - reg.real_auc) < GATE_TOLERANCE)
    result["passed"] = bool(ok)
    if not ok:
        raise GateFailure("seed-12 reproduction gate failed", result)
    return result


def choose_draws(s0_fit_seconds: float) -> dict:
    """The protocol's fallback, decided from the gate's timing alone."""
    def projected(b: int) -> float:
        return s0_fit_seconds * FITS_PER_DRAW * b * RUNTIME_OVERHEAD

    rec = {"s0_fit_seconds": s0_fit_seconds,
           "projected_seconds_b100": projected(B_DEFAULT),
           "projected_seconds_b50": projected(B_FALLBACK)}
    if projected(B_DEFAULT) <= B_DEFAULT_LIMIT_S:
        rec["B"] = B_DEFAULT
    elif projected(B_FALLBACK) <= B_FALLBACK_LIMIT_S:
        rec["B"] = B_FALLBACK
    else:
        raise BudgetStop("projected runtime for B = 50 exceeds 12 h; stopped before any new fit",
                         rec)
    return rec


# --- the diagnostic ---------------------------------------------------------

def _scored(rep, y: np.ndarray, what: str) -> tuple[np.ndarray, float]:
    """The fit's out-of-fold scores, with its AUC re-derived and checked."""
    oof = np.asarray(rep.oof, dtype=float)
    if oof.shape != y.shape:
        raise RunFailure(f"{what}: out-of-fold scores have shape {oof.shape}, labels {y.shape}")
    valid = np.isfinite(oof)
    a = auc(y[valid], oof[valid])
    if not np.isfinite(a):
        raise RunFailure(f"{what}: AUC undefined (n_scored={int(valid.sum())})")
    if not (np.isfinite(rep.auc) and abs(a - rep.auc) <= 1e-12):
        raise RunFailure(f"{what}: re-derived AUC {a} != reported {rep.auc}")
    return oof, a


@dataclass
class DrawLabels:
    donors_full: dict[str, str]
    donors_subset: dict[str, str]
    donors_strata: dict[str, str]
    y: dict[str, np.ndarray]           # arm -> labels on that arm's sample


def draw_labels(full: F6Inputs, sub: F6Inputs, b: int) -> DrawLabels:
    """Every labelling of draw b. S0 and S1 share one derangement on each sample."""
    dfull, dsub, dstr = full_donors(full.order, b), subset_donors(sub.order, b), \
        stratum_donors(sub.lengths, b)
    return DrawLabels(dfull, dsub, dstr, {
        "S0": shuffled_labels(full, dfull, "S0"),
        "S1": shuffled_labels(full, dfull, "S1"),
        "sub_S0": shuffled_labels(sub, dsub, "S0"),
        "sub_S1": shuffled_labels(sub, dsub, "S1"),
        "sub_S2": shuffled_labels(sub, dstr, "S2"),
    })


def stage0(full: F6Inputs, sub: F6Inputs, labels: Sequence[DrawLabels]) -> dict:
    out: dict[str, Any] = {"original_full": label_statistics(full, full.y, None, None),
                           "original_subset": label_statistics(sub, sub.y, None, None),
                           "draws": []}
    for b, dl in enumerate(labels):
        out["draws"].append({
            "draw": b,
            "S0": label_statistics(full, dl.y["S0"], dl.donors_full, "S0"),
            "S1": label_statistics(full, dl.y["S1"], dl.donors_full, "S1"),
            "sub_S0": label_statistics(sub, dl.y["sub_S0"], dl.donors_subset, "S0"),
            "sub_S1": label_statistics(sub, dl.y["sub_S1"], dl.donors_subset, "S1"),
            "sub_S2": label_statistics(sub, dl.y["sub_S2"], dl.donors_strata, "S2"),
        })
    return out


def fit_draw(full: F6Inputs, sub: F6Inputs, dl: DrawLabels, probe_cfg, cv_fn: Callable,
             b: int) -> tuple[dict, dict]:
    """Five activation fits, two position-feature fits, two oracles for draw b."""
    rec: dict[str, Any] = {"draw": b}
    scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm, inp in (("S0", full), ("S1", full), ("sub_S0", sub), ("sub_S1", sub),
                     ("sub_S2", sub)):
        y = dl.y[arm]
        oof, a = _scored(cv_fn(inp.X, y, inp.groups, probe_cfg), y, f"draw {b} probe {arm}")
        valid = np.isfinite(oof)
        rec[f"probe_{arm}"] = {"auc": a, "n_scored": int(valid.sum()),
                               "n_unscored": int((~valid).sum())}
        scores[f"probe_{arm}"] = (y, oof)
    for arm in ("S0", "S1"):
        y = dl.y[arm]
        oof, a = _scored(cv_fn(full.P, y, full.groups, probe_cfg), y, f"draw {b} P {arm}")
        valid = np.isfinite(oof)
        rec[f"P_{arm}"] = {"auc": a, "n_scored": int(valid.sum()),
                           "n_unscored": int((~valid).sum())}
        scores[f"P_{arm}"] = (y, oof)
        g = oracle_scores(full, dl.donors_full, arm)
        probe_valid = np.isfinite(scores[f"probe_{arm}"][1])
        g_masked = np.where(probe_valid, g, np.nan)    # rows the probe left unscored drop out
        oa = auc(y[probe_valid], g[probe_valid])
        if not np.isfinite(oa):
            raise RunFailure(f"draw {b}: oracle AUC undefined for {arm}")
        rec[f"oracle_{arm}"] = {"auc": oa, "n_scored": int(probe_valid.sum())}
        scores[f"oracle_{arm}"] = (y, g_masked)
    return rec, scores


#: name -> (sample, kind, arms): a level is one arm's AUC; a difference is
#: first minus second; a ratio is (first - 0.5) / (second - 0.5).
STATISTICS = {
    "probe_S0": ("full", "level", ("probe_S0",)),
    "probe_S1": ("full", "level", ("probe_S1",)),
    "oracle_S0": ("full", "level", ("oracle_S0",)),
    "oracle_S1": ("full", "level", ("oracle_S1",)),
    "P_S0": ("full", "level", ("P_S0",)),
    "P_S1": ("full", "level", ("P_S1",)),
    "delta_A": ("full", "difference", ("probe_S1", "probe_S0")),
    "E_S0": ("full", "difference", ("probe_S0", "oracle_S0")),
    "E_S1": ("full", "difference", ("probe_S1", "oracle_S1")),
    "share_of_oracle_S0": ("full", "ratio", ("probe_S0", "oracle_S0")),
    "share_of_oracle_S1": ("full", "ratio", ("probe_S1", "oracle_S1")),
    "subset_probe_S0": ("subset", "level", ("probe_sub_S0",)),
    "subset_probe_S1": ("subset", "level", ("probe_sub_S1",)),
    "subset_probe_S2": ("subset", "level", ("probe_sub_S2",)),
    "subset_S2_minus_S1": ("subset", "difference", ("probe_sub_S2", "probe_sub_S1")),
    "subset_S2_minus_S0": ("subset", "difference", ("probe_sub_S2", "probe_sub_S0")),
    "subset_S1_minus_S0": ("subset", "difference", ("probe_sub_S1", "probe_sub_S0")),
}


def _combine(kind: str, a: np.ndarray, b: np.ndarray | None) -> np.ndarray:
    if kind == "level":
        return a
    if kind == "difference":
        return a - b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b != 0.5, (a - 0.5) / (b - 0.5), np.nan)


def summarise(per_draw: list[dict], scores: list[dict], samples: dict[str, F6Inputs],
              n_boot: int, seed: int) -> dict:
    """Both uncertainty summaries for every statistic, never combined."""
    counts = {name: bootstrap_counts(len(inp.order), n_boot, seed)
              for name, inp in samples.items()}
    arm_sample = {k: v[0] for v in STATISTICS.values() for k in v[2]}
    boot: dict[str, np.ndarray] = {}            # arm -> (n_boot, B) AUC
    for arm, sample in arm_sample.items():
        inp = samples[sample]
        cols = []
        for sc in scores:
            y, s = sc[arm]
            v = np.isfinite(s)
            cols.append(weighted_auc(y[v], s[v], inp.problem_of_row[v], counts[sample]))
        boot[arm] = np.stack(cols, axis=1)
    out: dict[str, Any] = {}
    for name, (sample, kind, arms) in STATISTICS.items():
        a = np.array([d[f"{arms[0]}"]["auc"] for d in per_draw])
        b = np.array([d[f"{arms[1]}"]["auc"] for d in per_draw]) if len(arms) > 1 else None
        values = _combine(kind, a, b)
        rep = _combine(kind, boot[arms[0]], boot[arms[1]] if len(arms) > 1 else None)
        rep_mean = np.where(np.all(np.isfinite(rep), axis=1), np.mean(rep, axis=1), np.nan)
        entry = {"sample": sample, "kind": kind, "arms": list(arms),
                 "donor_draws": draw_summary(values),
                 "problem_bootstrap": bootstrap_summary(rep_mean)}
        if kind == "difference":
            entry["wording"] = wording(entry["donor_draws"], entry["problem_bootstrap"])
        out[name] = entry
    return out


def run_diagnostic(full: F6Inputs, data: dict, paths: Paths, probe_cfg, provenance: dict,
                   cv_fn: Callable, reg: Registered, analysis_seed: int,
                   draws_fn: Callable[[float], dict] = choose_draws,
                   n_boot: int = N_BOOTSTRAP, max_draws: int = B_DEFAULT) -> dict:
    """Gate, then the draws, then the summaries. Output only after the gate."""
    if paths.out_dir.exists():
        raise PreflightFailure(f"{paths.out_dir} exists; refusing to run")
    check_robustness(paths, "before the run")
    eligible, _ = exact_length_eligible(full.lengths)
    sub = full.subset(eligible)
    labels = [draw_labels(full, sub, b) for b in range(max_draws)]     # before any fit
    stage0_all = stage0(full, sub, labels)

    gate = run_gate(full, probe_cfg, cv_fn, reg)
    budget = draws_fn(gate["s0_fit_seconds"])
    B = int(budget["B"])

    try:
        paths.out_dir.mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        raise PreflightFailure(f"{paths.out_dir} exists; refusing to run") from None
    started = datetime.now(timezone.utc).isoformat()
    status = paths.out_dir / "status.json"
    per_draw: list[dict] = []
    scores: list[dict] = []
    stage = "outputs"
    try:
        write_json(status, {"_label": LABEL, "status": "running", "started": started})
        write_json(paths.out_dir / "provenance.json", {
            "_label": LABEL, **provenance, "gate": gate, "draw_budget": budget,
            "data_checks": data, "started": started})
        write_json(paths.out_dir / "stage0_label_stats.json", {
            "_label": LABEL, "computed_before_any_fit": True, "B": B,
            **{k: v for k, v in stage0_all.items() if k != "draws"},
            "draws": stage0_all["draws"][:B]})
        stage = "draws"
        with open(paths.out_dir / "draws.jsonl", "w", encoding="utf-8") as fh:
            for b in range(B):
                rec, sc = fit_draw(full, sub, labels[b], probe_cfg, cv_fn, b)
                if b == 0 and abs(rec["probe_S0"]["auc"] - gate["shuffled_auc"]) > 1e-12:
                    raise RunFailure("draw 0 S0 does not equal the gate's registered fit")
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                per_draw.append(rec)
                scores.append(sc)
        stage = "summaries"
        summary = summarise(per_draw, scores, {"full": full, "subset": sub}, n_boot,
                            analysis_seed)
        np.savez_compressed(paths.out_dir / "oof_scores.npz",
                            **{f"d{b}_{k}": s for b, sc in enumerate(scores)
                               for k, (_, s) in sc.items()})
        after = check_robustness(paths, "after the run")
        write_json(paths.out_dir / "summary.json", {
            "_label": LABEL, "B": B, "n_bootstrap": n_boot, "bootstrap_seed": analysis_seed,
            "caveats": CAVEATS,
            "reading": "see the protocol's reading rules; this output gives no leakage verdict",
            "statistics": summary})
        write_json(status, {"_label": LABEL, "status": "complete", "started": started,
                            "finished": datetime.now(timezone.utc).isoformat(),
                            "B": B, "n_draws_completed": len(per_draw),
                            "robustness_sha256_after": after})
        return {"status": "complete", "B": B, "n_draws_completed": len(per_draw)}
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised as a failure
        try:
            after = sha256_file(paths.robustness)
        except OSError as e:
            after = f"unreadable: {e}"
        write_json(status, {"_label": LABEL, "status": "failed", "started": started,
                            "failed_stage": stage, "n_draws_planned": B,
                            "n_draws_completed": len(per_draw),
                            "failed_draw": len(per_draw) if stage == "draws" else None,
                            "error": repr(exc), "traceback": traceback.format_exc(),
                            "robustness_sha256_after": after,
                            "robustness_unchanged": after == ROBUSTNESS_SHA256})
        if isinstance(exc, DiagnosticStop):
            raise
        raise RunFailure(f"run failed ({stage}, {len(per_draw)} draws done): {exc!r}") from exc


# --- entry point --------------------------------------------------------------

def config_values(cfg) -> dict:
    return {"probe": {"kind": cfg.probe.kind, "C": float(cfg.probe.C),
                      "max_iter": int(cfg.probe.max_iter),
                      "standardise": bool(cfg.probe.standardise),
                      "n_folds": int(cfg.probe.n_folds), "seed": int(cfg.probe.seed)},
            "analysis_seed": int(cfg.analysis.seed), "layer": int(cfg.target.layer),
            "config_hash": cfg.hash()}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["preflight", "gate", "run"], required=True)
    args = ap.parse_args(argv)

    cfg = config_mod.load(SOURCE_RUN, [])
    paths = Paths.default(cfg)
    reg = Registered()
    cv = config_values(cfg)
    report: dict[str, Any] = {"_label": LABEL, "phase": args.phase}
    try:
        report["robustness_sha256_before"] = check_robustness(paths, "before the run")
        report["preflight"] = preflight(paths, cv)
        full, extra = load_f6_inputs(cfg)
        report["data_checks"] = data_checks(full, extra, reg)
        if args.phase == "gate":
            report["gate"] = run_gate(full, cfg.probe, probe_mod.cross_validate, reg)
            try:                               # informational here; `run` enforces it
                report["draw_budget"] = choose_draws(report["gate"]["s0_fit_seconds"])
            except BudgetStop as stop:
                report["draw_budget"] = {"stop": stop.reason, **stop.details}
        elif args.phase == "run":
            prov = {"protocol_sha256": PROTOCOL_SHA256,
                    "script_sha256_lf": sha256_text_lf(paths.script),
                    "robustness_sha256_before": report["robustness_sha256_before"],
                    "source_fingerprints": SOURCE_FINGERPRINTS, "config": cv,
                    "git": report["preflight"]["script_protocol_decisions_committed"]["value"],
                    "versions": {"python": platform.python_version(),
                                 "numpy": np.__version__}}
            report["result"] = run_diagnostic(full, report["data_checks"], paths, cfg.probe,
                                              prov, probe_mod.cross_validate, reg,
                                              cv["analysis_seed"])
        report["robustness_sha256_after"] = check_robustness(paths, "after the run")
        report["status"] = "ok"
        code = 0
    except DiagnosticStop as stop:
        report.update({"status": type(stop).__name__, "reason": stop.reason,
                       "details": stop.details})
        try:
            report["robustness_sha256_after"] = sha256_file(paths.robustness)
        except OSError:
            report["robustness_sha256_after"] = None
        code = stop.code
    print(json.dumps(report, indent=2, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())
