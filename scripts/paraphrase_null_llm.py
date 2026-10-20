"""A rewording null for RQ2: model-written paraphrases, QC'd and scored (D76). GPU.

D70 found the registered paraphrase null is mostly no edit (29%) or an added
discourse marker (41%). It cannot show how far reconstruction moves under a
*rewording* that keeps the meaning. This study generates rewordings with the
target model, filters them by rules fixed in D76 before generation, and scores
them with the reconstructor. **Exploratory**: no paraphrase's meaning
preservation is validated, because that needs the human sheet this script
writes. The registered RQ2 result is unchanged.

Phases, each resumable, each loading one 7B model:

1. ``--phase generate``: the target model rewrites each claim of a seeded,
   stratified sample of explanations (3 variants per claim, T = 0.7).
2. ``--phase score``: the reconstructor scores each explanation with one claim
   replaced by an accepted rewording. It also re-scores every base
   explanation, and the run stops if those disagree with the stored base
   cosines (checks that this scoring path reproduces the registered one).
   Texts identical to an already scored text are scored once.
3. ``--phase analyse`` (CPU): the D76 comparisons.

    python scripts/paraphrase_null_llm.py --config main --phase generate
    python scripts/paraphrase_null_llm.py --config main --phase score
    python scripts/paraphrase_null_llm.py --config main --phase analyse

Writes ``runs/<run_id>_paraphrase_llm/``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time

import numpy as np

from _stage import base_parser, load_config

from nlaast.analysis.posthoc import cluster_bootstrap_mean
from nlaast.faithfulness.claims import split_claims, substitute_claim
from nlaast.faithfulness.paraphrase_qc import (
    IDENTICAL,
    NORMALISED_IDENTICAL,
    classify,
    token_change,
)
from nlaast.logging_utils import JsonlWriter, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow

SEED = 20261019
N_EXPLANATIONS = 60
N_VARIANTS = 3
TEMPERATURE = 0.7
MAX_NEW_TOKENS = 120
#: QC rules, fixed in D76 before any paraphrase was generated.
MIN_TOKEN_CHANGE = 0.25
LENGTH_RATIO = (0.6, 1 / 0.6)
BASE_COSINE_TOL = 0.01
INSTRUCTION = ("Rewrite the following sentence so that it means exactly the same thing "
               "but uses different wording. Keep every number, every quoted string and "
               "every technical term exactly as written. Reply with the rewritten "
               "sentence only.\n\nSentence: ")

_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_QUOTED = re.compile(r"[\"“]([^\"”]+)[\"”]")


def qc(original: str, variant: str) -> tuple[bool, str]:
    """D76's acceptance rule for one rewording. Returns (accepted, reason)."""
    v = variant.strip()
    if not v or "\n" in v:
        return False, "empty or multi-line"
    cls = classify(original.strip(), v)
    if cls in (IDENTICAL, NORMALISED_IDENTICAL):
        return False, cls
    if sorted(_NUM.findall(original)) != sorted(_NUM.findall(v)):
        return False, "numbers changed"
    if any(q not in v for q in _QUOTED.findall(original)):
        return False, "quoted string changed"
    lo, hi = LENGTH_RATIO
    ratio = len(v.split()) / max(1, len(original.split()))
    if not lo <= ratio <= hi:
        return False, "length ratio"
    if token_change(original, v) < MIN_TOKEN_CHANGE:
        return False, "too close to the original"
    return True, "accepted"


def sample_explanations(audits: list[dict], n: int, seed: int) -> list[dict]:
    """Seeded sample, stratified by window kind, of explanations with claims."""
    rng = np.random.default_rng(seed)
    by_kind: dict[str, list[dict]] = {}
    for a in audits:
        if a.get("claims"):
            by_kind.setdefault(a["window_kind"], []).append(a)
    out = []
    per = n // len(by_kind)
    for k in sorted(by_kind):
        pool = sorted(by_kind[k], key=lambda a: (a["problem_id"], a.get("sample_index", 0),
                                                 a.get("chunk_index", 0)))
        out += [pool[int(i)] for i in rng.choice(len(pool), size=min(per, len(pool)),
                                                 replace=False)]
    return out


def audit_key(a: dict) -> str:
    return f"{a['problem_id']}|{a['window_kind']}|{a.get('chunk_index')}|{a.get('sample_index')}"


def phase_generate(cfg, out_dir, log):
    from nlaast.models.target import TargetModel

    audits = read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")
    sample = sample_explanations(audits, N_EXPLANATIONS, SEED)
    write_json(out_dir / "sample.json", {"seed": SEED, "explanations": [audit_key(a) for a in sample]})
    model = TargetModel(cfg)
    writer = JsonlWriter(out_dir / "paraphrases.jsonl", key="id")
    started, n = time.monotonic(), 0
    try:
        with writer:
            for a in sample:
                for c in a["claims"]:
                    key = f"{audit_key(a)}|{c['claim_index']}"
                    if writer.has(key):
                        continue
                    prompt = model.build_prompt(INSTRUCTION + c["claim_text"])
                    gens = model.generate([prompt] * N_VARIANTS, max_new_tokens=MAX_NEW_TOKENS,
                                          temperature=TEMPERATURE, top_p=0.95, do_sample=True,
                                          seed=SEED + n, batch_size=N_VARIANTS)
                    n += 1
                    variants = []
                    for g in gens:
                        text = g.text.strip().strip('"').strip()
                        ok, why = qc(c["claim_text"], text)
                        variants.append({"text": text, "accepted": ok, "reason": why,
                                         "token_change": token_change(c["claim_text"], text)})
                    writer.write({"id": key, "audit": audit_key(a), "claim_index": c["claim_index"],
                                  "claim_text": c["claim_text"], "variants": variants})
    finally:
        model.close()
    return {"claims": n, "seconds": time.monotonic() - started}


def phase_score(cfg, out_dir, log):
    from nlaast.activations.store import ActivationStore
    from nlaast.models import loading
    from nlaast.nla.reconstructor import ActivationReconstructor

    audits = {audit_key(a): a for a in read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")}
    verbs = {f"{v['problem_id']}|{v['window_kind']}|{v['chunk_index']}": v
             for v in read_jsonl(cfg.dir / "nla" / "verbalisations.jsonl")}
    paras = read_jsonl(out_dir / "paraphrases.jsonl")
    store = ActivationStore(cfg.dir / "acts")
    ar = ActivationReconstructor(cfg, loading.ensure_nf4(cfg.nla.ar_repo).local_path)
    writer = JsonlWriter(out_dir / "scores.jsonl", key="id")
    cache: dict[tuple[str, str], float] = {}
    started, passes, reused = time.monotonic(), 0, 0

    def cosine(text, gold, akey):
        nonlocal passes, reused
        k = (akey, text)
        if k not in cache:
            cache[k] = float(ar.score(text, gold).cosine)
            passes += 1
        else:
            reused += 1
        return cache[k]

    try:
        with writer:
            base_checks = []
            for akey in sorted({p["audit"] for p in paras}):
                a = audits[akey]
                v = verbs[f"{a['problem_id']}|{a['window_kind']}|{a['chunk_index']}"]
                gold = store.get(a["problem_id"], cfg.target.layer)[0][v["activation_row"]]
                base = cosine(a["explanation"], gold, akey)
                base_checks.append(abs(base - a["base_cosine"]))
                claims = {c.index: c for c in split_claims(a["explanation"],
                                                           cfg.faithfulness.max_claims_per_explanation)}
                for p in [p for p in paras if p["audit"] == akey]:
                    c = claims[p["claim_index"]]
                    for j, var in enumerate(p["variants"]):
                        if not var["accepted"]:
                            continue
                        key = f"{p['id']}|v{j}"
                        if writer.has(key):
                            continue
                        cos = cosine(substitute_claim(a["explanation"], c, var["text"]), gold, akey)
                        writer.write({"id": key, "audit": akey, "claim_index": p["claim_index"],
                                      "variant": j, "cosine": cos, "effect": base - cos,
                                      "base_cosine": base})
            if max(base_checks) > BASE_COSINE_TOL:
                raise SystemExit(f"re-scored base cosines differ from the stored ones by up to "
                                 f"{max(base_checks):.4f} (> {BASE_COSINE_TOL}); not comparable")
    finally:
        ar.close()
    return {"ar_passes": passes, "reused_identical": reused,
            "seconds": time.monotonic() - started,
            "max_base_cosine_diff": max(base_checks) if base_checks else None}


def phase_analyse(cfg, out_dir):
    audits = {audit_key(a): a for a in read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")}
    paras = read_jsonl(out_dir / "paraphrases.jsonl")
    scores = read_jsonl(out_dir / "scores.jsonl")
    variants = [v for p in paras for v in p["variants"]]
    reasons: dict[str, int] = {}
    for v in variants:
        reasons[v["reason"]] = reasons.get(v["reason"], 0) + 1
    eff_by_audit: dict[str, list[float]] = {}
    eff_by_claim: dict[tuple[str, int], list[float]] = {}
    for s in scores:
        eff_by_audit.setdefault(s["audit"], []).append(abs(s["effect"]))
        eff_by_claim.setdefault((s["audit"], s["claim_index"]), []).append(s["effect"])
    pct = cfg.faithfulness.null_percentile
    rows = []
    for akey in sorted({p["audit"] for p in paras}):
        a = audits[akey]
        thr_llm = (float(np.percentile(eff_by_audit[akey], pct))
                   if eff_by_audit.get(akey) else None)
        for c in a["claims"]:
            rows.append({
                "problem_id": a["problem_id"], "audit": akey,
                "delete_effect": c["delete_effect"],
                "registered_dependent": bool(c["reconstruction_dependent"]),
                "registered_threshold": c["null_threshold"],
                "rule_paraphrase_mean": float(np.mean(c["paraphrase_effects"])),
                "llm_threshold": thr_llm,
                "llm_mean_effect": (float(np.mean(eff_by_claim[(akey, c["claim_index"])]))
                                    if (akey, c["claim_index"]) in eff_by_claim else None),
            })
    ev = [r for r in rows if r["llm_threshold"] is not None]
    rng = np.random.default_rng(SEED)
    pid = [r["problem_id"] for r in ev]
    dep_llm = [float(r["delete_effect"] > r["llm_threshold"]) for r in ev]
    dep_reg = [float(r["registered_dependent"]) for r in ev]
    paired = [r for r in ev if r["llm_mean_effect"] is not None]
    rule_abs = [abs(r["rule_paraphrase_mean"]) for r in paired]
    llm_abs = [abs(r["llm_mean_effect"]) for r in paired]
    out = {
        "status": "exploratory (D76); meaning preservation of the rewordings is unvalidated",
        "n_explanations": len({r["audit"] for r in rows}), "n_claims": len(rows),
        "n_claims_evaluable": len(ev), "n_variants": len(variants), "qc_outcomes": reasons,
        "acceptance_rate": reasons.get("accepted", 0) / len(variants) if variants else None,
        "median_threshold_registered": float(np.median([r["registered_threshold"] for r in ev])),
        "median_threshold_rewording": float(np.median([r["llm_threshold"] for r in ev])),
        "dependent_rate_registered_null": cluster_bootstrap_mean(dep_reg, pid, 10000, rng),
        "dependent_rate_rewording_null": cluster_bootstrap_mean(dep_llm, pid, 10000, rng),
        "mean_abs_effect_rule_paraphrase": float(np.mean(rule_abs)) if rule_abs else None,
        "mean_abs_effect_rewording": float(np.mean(llm_abs)) if llm_abs else None,
        "mean_delete_effect": float(np.mean([r["delete_effect"] for r in paired])) if paired else None,
        "share_delete_above_rewording": (float(np.mean([r["delete_effect"] > abs(r["llm_mean_effect"])
                                                        for r in paired])) if paired else None),
    }
    # human validation sheet for the accepted rewordings (blank)
    with open(out_dir / "rewording_validation_sheet.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "original", "rewording", "same_meaning", "note"])
        w.writeheader()
        for p in paras:
            for j, v in enumerate(p["variants"]):
                if v["accepted"]:
                    w.writerow({"id": f"{p['id']}|v{j}", "original": p["claim_text"],
                                "rewording": v["text"], "same_meaning": "", "note": ""})
    return out


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--phase", choices=["generate", "score", "analyse"], required=True)
    args = p.parse_args()
    cfg = load_config(args)
    out_dir = cfg.dir.parent / f"{cfg.run_id}_paraphrase_llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    import logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("paraphrase_llm")
    if args.phase == "generate":
        res = phase_generate(cfg, out_dir, log)
    elif args.phase == "score":
        res = phase_score(cfg, out_dir, log)
    else:
        res = phase_analyse(cfg, out_dir)
    res.update({"phase": args.phase, "created": utcnow(), "git": git_commit()})
    write_json(out_dir / f"{args.phase}.json", res)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
