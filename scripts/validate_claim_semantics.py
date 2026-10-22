"""Objective checks of what each verbalisation says about its context (D71). CPU only.

For every activation-driven explanation in the NLA arm that passed the
integrity gate, checks its final-token claim, its quoted context and its
expected continuation against the stored trace. Each is also checked under
three controls: a mismatched trace, a shifted position, and the Gaussian
control's explanation. The analysis plan is fixed in D71. Post hoc: none of
this is a registered result.

    python scripts/validate_claim_semantics.py --config main

Writes ``runs/<run_id>_claim_checks/claim_checks.json`` and refuses to replace
it (``--force``).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from _stage import base_parser, load_config

from nlaast.analysis.posthoc import cluster_bootstrap_mean
from nlaast.faithfulness.claim_checks import (
    expectation_hit,
    final_token_match,
    parse,
    quote_support,
)
from nlaast.logging_utils import read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

SEED = 20261019
REPS = 10000
CONTINUATION_CHARS = 200
SUPPORTED = 0.8


def position_text(trace: dict, chunk: int) -> tuple[str, str]:
    """(context up to and including the activation's token, what follows)."""
    tok = trace["chunks"][chunk]["token_end"]
    end = trace["offsets"][tok][1]
    ctx = trace["question"] + "\n\n" + trace["trace"][:end]
    return ctx, trace["trace"][end:end + CONTINUATION_CHARS]


def score(explanation: str, ctx: str, cont: str) -> dict[str, float | None]:
    p = parse(explanation)
    out: dict[str, float | None] = {"has_final_token_claim": float(p.final_token is not None),
                                    "mechanistic_language": float(p.mechanistic)}
    if p.final_token is not None:
        out.update({f"final_{k}": float(v) for k, v in final_token_match(p.final_token, ctx).items()})
    sup = [s for s in (quote_support(q, ctx) for q in p.context_quotes) if s is not None]
    if sup:
        out["quote_supported_share"] = float(np.mean([s >= SUPPORTED for s in sup]))
        out["quote_mean_support"] = float(np.mean(sup))
    hit = expectation_hit(p.expected, cont)
    if hit is not None:
        out.update({f"expect_{k}": float(v) for k, v in hit.items()})
    return out


METRICS = ("final_exact_suffix", "final_last_word", "final_last_char",
           "quote_supported_share", "quote_mean_support",
           "expect_first_word", "expect_any_word_in_first_10")


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args)
    path = cfg.dir.parent / f"{cfg.run_id}_claim_checks" / "claim_checks.json"
    if path.exists() and not args.force:
        print(f"{path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    verbs = read_jsonl(cfg.dir / "nla" / "verbalisations.jsonl")
    pids = sorted({v["problem_id"] for v in verbs})
    nxt = {p: pids[(i + 1) % len(pids)] for i, p in enumerate(pids)}

    rows = []
    for v in verbs:
        pid, c = v["problem_id"], int(v["chunk_index"])
        t = traces[pid]
        n = len(t["chunks"])
        ctx, cont = position_text(t, c)
        other = traces[nxt[pid]]
        c_other = int(round(c / max(1, n - 1) * (len(other["chunks"]) - 1)))
        octx, ocont = position_text(other, c_other)
        c_shift = (c + n // 2) % n
        sctx, scont = position_text(t, c_shift)
        noise = [s["explanation"] for s in v.get("noise_samples", []) if s.get("ok")]
        for s in v["samples"]:
            if not s.get("ok"):
                continue
            e = s["explanation"]
            rows.append({
                "problem_id": pid, "window_kind": v["window_kind"], "chunk_index": c,
                "real": score(e, ctx, cont),
                "mismatched_trace": score(e, octx, ocont),
                "shifted_position": score(e, sctx, scont) if c_shift != c else None,
                "gaussian_control": score(noise[0], ctx, cont) if noise else None,
            })

    rng = np.random.default_rng(SEED)
    results = {}
    for m in METRICS:
        real = [(r["problem_id"], r["real"].get(m)) for r in rows if r["real"].get(m) is not None]
        entry = {"n_explanations": len(real),
                 "real_rate": float(np.mean([x for _, x in real])) if real else None,
                 "controls": {}}
        for ctrl in ("mismatched_trace", "shifted_position", "gaussian_control"):
            paired = [(r["problem_id"], r["real"][m] - r[ctrl][m]) for r in rows
                      if r[ctrl] is not None and r["real"].get(m) is not None
                      and r[ctrl].get(m) is not None]
            ctrl_vals = [r[ctrl][m] for r in rows if r[ctrl] is not None and r[ctrl].get(m) is not None]
            if not paired:
                entry["controls"][ctrl] = None
                continue
            d = cluster_bootstrap_mean([x for _, x in paired], [p for p, _ in paired], REPS, rng)
            entry["controls"][ctrl] = {"control_rate": float(np.mean(ctrl_vals)),
                                       "n_paired": len(paired),
                                       "real_minus_control": d,
                                       "excludes_zero_positive": d["low"] > 0}
        entry["content_specific"] = (
            all(v is not None and v["excludes_zero_positive"] for v in entry["controls"].values()))
        results[m] = entry
    out = {
        "status": "post hoc, objective checks (D71); no registered result",
        "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(), "seed": SEED,
        "n_explanations": len(rows), "n_problems": len({r["problem_id"] for r in rows}),
        "has_final_token_claim": float(np.mean([r["real"]["has_final_token_claim"] for r in rows])),
        "mechanistic_language_rate": float(np.mean([r["real"]["mechanistic_language"] for r in rows])),
        "metrics": results,
        "by_window_kind": {
            k: {m: float(np.mean(vals)) if (vals := [r["real"][m] for r in rows
                                                       if r["window_kind"] == k and r["real"].get(m) is not None]) else None
                for m in METRICS}
            for k in ("tail", "matched_position", "matched_length")},
    }
    write_json(path, out)
    print(json.dumps(out, indent=1))
    print("wrote", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
