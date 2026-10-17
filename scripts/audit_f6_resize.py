"""Diagnostic audit of F6's across-groups label shuffle (separate from F6).

F6 (``run_falsification_tests.f6_probe_leakage``) permutes whole problems'
label sequences between problems with ``baselines.probe.permute_labels``,
fitting each donor sequence to the receiver's length with ``np.resize`` -
tiling a shorter donor from its start and truncating a longer one. This script
reproduces F6's labels, groups and shuffle exactly (same assembly, same
function, same seed) and measures what that does to the labels:

* how many receivers get a tiled or truncated donor, and how many labels are
  repeated or dropped;
* whether the permutation leaves any problem with its own labels;
* prevalence, overall and per problem, before and after;
* how strongly the labels are tied to boundary position before and after -
  the property that let a positional shuffle masquerade as leakage in RQ2.

**Label statistics only.** No probe is fitted, F6 is not rerun, and nothing
here changes F6's code, result or interpretation. CPU-only (no torch import).

    python scripts/audit_f6_resize.py --config main
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

from _stage import base_parser

from nlaast import config as config_mod
from nlaast.activations.store import ActivationStore
from nlaast.baselines.probe import permute_labels
from nlaast.data.answers import PERMISSIVE, equivalent, extract_answer
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.trace.chunking import Chunk, prefix_text


def f6_labels(cfg):
    """F6's (y, groups, relative positions), assembled exactly as F6 does."""
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    store = ActivationStore(cfg.dir / "acts")
    layer = cfg.target.layer
    y, groups, rel = [], [], []
    for pid, t in traces.items():
        if not store.has(pid, layer):
            continue
        _, meta = store.get(pid, layer, verify=False)
        chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                             "token_end", "prefix_tokens")})
                  for c in t.get("chunks", [])]
        n = len(chunks)
        for ci in meta.get("chunk_indices", []):
            if ci >= n:
                continue
            ans = extract_answer(prefix_text(chunks, ci, t["trace"]), PERMISSIVE)
            if ans is None:
                continue
            y.append(1 if equivalent(ans, t.get("gold")) else 0)
            groups.append(pid)
            rel.append(ci / (n - 1) if n > 1 else 1.0)
    return np.asarray(y, dtype=int), groups, np.asarray(rel)


def rank_auc(score: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import mannwhitneyu
    pos, neg = score[y == 1], score[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(mannwhitneyu(pos, neg).statistic / (len(pos) * len(neg)))


def donor_map(groups, seed):
    """Recover the donor of each receiver by re-running permute_labels' draw."""
    order = sorted(set(groups))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(order))
    if len(order) > 1:
        for _ in range(10):
            if all(perm[i] != i for i in range(len(order))):
                break
            perm = rng.permutation(len(order))
    return {g: order[p] for g, p in zip(order, perm)}


def audit(cfg) -> dict:
    y, groups, rel = f6_labels(cfg)
    seed = cfg.probe.seed + 1                     # what cross_validate passes
    y_shuf, _ = permute_labels(y, groups, "across_groups", seed)
    lengths = Counter(groups)
    donors = donor_map(groups, seed)

    # Check the reconstruction of permute_labels' internals against its output.
    by_g = {g: y[[i for i, h in enumerate(groups) if h == g]] for g in lengths}
    rebuilt = np.empty_like(y)
    for g in lengths:
        idx = [i for i, h in enumerate(groups) if h == g]
        rebuilt[idx] = np.resize(by_g[donors[g]], len(idx))
    assert np.array_equal(rebuilt, y_shuf), "donor reconstruction does not match permute_labels"

    tiled = [g for g in lengths if lengths[donors[g]] < lengths[g]]
    truncated = [g for g in lengths if lengths[donors[g]] > lengths[g]]
    equal = [g for g in lengths if lengths[donors[g]] == lengths[g]]
    repeated = sum(lengths[g] - lengths[donors[g]] for g in tiled)
    dropped = sum(lengths[donors[g]] - lengths[g] for g in truncated)
    # Labels a receiver gets from a donor's *later* boundaries are lost when
    # truncated; tiling restarts the donor at its first boundary.
    dropped_pos = sum(int(by_g[donors[g]][lengths[g]:].sum()) for g in truncated)
    tiled_restart_pos = sum(int(np.resize(by_g[donors[g]], lengths[g])[lengths[donors[g]]:].sum())
                            for g in tiled)

    def per_problem_prev(v):
        return {g: float(np.mean(v[[i for i, h in enumerate(groups) if h == g]])) for g in lengths}

    pp0, pp1 = per_problem_prev(y), per_problem_prev(y_shuf)
    donor_prev = {g: pp0[donors[g]] for g in lengths}
    prev_shift = [abs(pp1[g] - donor_prev[g]) for g in lengths]

    # Monotone structure: once an intermediate answer is right, does it stay right?
    def monotone_share(v):
        ok = 0
        for g in lengths:
            s = v[[i for i, h in enumerate(groups) if h == g]]
            ok += int(np.all(np.diff(s) >= 0))
        return ok / len(lengths)

    stored = read_json(cfg.dir / "robustness" / "robustness.json")["F6_probe_leakage"]
    return {
        "scope": "diagnostic audit of F6's across_groups shuffle; label statistics only, "
                 "no probe fitted, F6 not rerun or modified",
        "reproduces_registered_shuffle": {
            "n_rows": int(len(y)), "n_problems": len(lengths),
            "n_labels_changed": int(np.sum(y != y_shuf)),
            "registered_n_labels_changed": stored["shuffles"]["across_groups"]["n_labels_changed"],
            "matches": int(np.sum(y != y_shuf)) == stored["shuffles"]["across_groups"]["n_labels_changed"],
        },
        "registered_result": {"real_auc": stored["real_auc"],
                              "shuffled_auc_across_groups": stored["shuffles"]["across_groups"]["auc"],
                              "shuffled_auc_within_group": stored["shuffles"]["within_group"]["auc"],
                              "leak_rule": "|shuffled AUC - 0.5| > 0.15",
                              "leak_suspected": stored["leak_suspected"]},
        "donor_assignment": {
            "fixed_points": sum(1 for g in lengths if donors[g] == g),
            "bijection": len(set(donors.values())) == len(lengths),
        },
        "length_matching": {
            "receivers_tiled": len(tiled), "receivers_truncated": len(truncated),
            "receivers_equal_length": len(equal),
            "labels_repeated_by_tiling": int(repeated),
            "labels_dropped_by_truncation": int(dropped),
            "positive_labels_dropped_by_truncation": dropped_pos,
            "positive_labels_in_tiled_restarts": tiled_restart_pos,
            "boundaries_per_problem": {"min": min(lengths.values()), "max": max(lengths.values()),
                                       "median": float(np.median(list(lengths.values())))},
        },
        "prevalence": {
            "original": float(y.mean()), "shuffled": float(y_shuf.mean()),
            "mean_abs_shift_vs_donor_prevalence_per_problem": float(np.mean(prev_shift)),
            "max_abs_shift_vs_donor_prevalence_per_problem": float(np.max(prev_shift)),
        },
        "position_association": {
            "auc_relative_position_original": rank_auc(rel, y),
            "auc_relative_position_shuffled": rank_auc(rel, y_shuf),
            "monotone_problem_share_original": monotone_share(y),
            "monotone_problem_share_shuffled": monotone_share(y_shuf),
        },
    }


def main() -> int:
    p = base_parser(__doc__)
    args = p.parse_args()
    cfg = config_mod.load(args.config, args.overrides)
    out = audit(cfg)
    path = cfg.dir.parent / f"{cfg.run_id}_f6_audit" / "f6_resize_audit.json"
    write_json(path, out)
    import json
    print(json.dumps(out, indent=2))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
