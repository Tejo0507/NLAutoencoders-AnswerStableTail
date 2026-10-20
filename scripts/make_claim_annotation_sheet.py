"""Blinded claim-annotation sheets for the D71 protocol, and scoring of returned ones.

Builds the sample described in ``docs/CLAIM_VALIDATION_PROTOCOL.md`` Part B:
60 activation-driven explanations, stratified by window kind and dataset and
split by problem into development and held-out sets. Ten Gaussian-control
explanations and ten explanations shown against a mismatched trace are mixed
in blind. The sheets contain **blank** label columns. Nothing here fills them,
and ``--score`` refuses a sheet with any blank label.

    python scripts/make_claim_annotation_sheet.py --config main
    python scripts/make_claim_annotation_sheet.py --config main --score a.csv b.csv

Writes ``runs/<run_id>_claim_annotation/{dev,heldout}.csv`` and ``key.json``;
refuses to replace them (``--force``).
"""

from __future__ import annotations

import csv
import sys

import numpy as np

from _stage import base_parser, load_config

from nlaast.faithfulness.claim_checks import cohens_kappa
from nlaast.faithfulness.claims import split_claims
from nlaast.logging_utils import read_jsonl, write_json

SEED = 20261019
N_REAL, N_DEV, N_NOISE, N_MISMATCH = 60, 20, 10, 10
FIELDS = ("P1_understandable", "P2_supported", "P3_next_or_here", "P5_mechanism")


def sample_items(verbs: list[dict], rng: np.random.Generator) -> list[dict]:
    """Stratified real items, then blind controls, as (kind, verbalisation, sample) rows."""
    real = [(v, s) for v in verbs for s in v["samples"] if s.get("ok")]
    strata: dict[tuple, list] = {}
    for v, s in real:
        key = (v["window_kind"], v["problem_id"].split("-")[0])
        strata.setdefault(key, []).append((v, s))
    keys = sorted(strata)
    picked = []
    per = int(np.ceil(N_REAL / len(keys)))
    for k in keys:
        pool = strata[k]
        for i in rng.choice(len(pool), size=min(per, len(pool)), replace=False):
            picked.append(("real", *pool[int(i)]))
    picked = picked[:N_REAL]
    noise = [(v, s) for v in verbs for s in v.get("noise_samples", []) if s.get("ok")]
    for i in rng.choice(len(noise), size=N_NOISE, replace=False):
        picked.append(("gaussian_control", *noise[int(i)]))
    for i in rng.choice(len(real), size=N_MISMATCH, replace=False):
        picked.append(("mismatched_trace", *real[int(i)]))
    return picked


def split_by_problem(items: list[dict], rng: np.random.Generator) -> set[str]:
    """Problems whose items go to development; the rest are held out."""
    pids = sorted({it["problem_id"] for it in items})
    order = list(rng.permutation(len(pids)))
    dev, n = set(), 0
    for i in order:
        if n >= N_DEV:
            break
        p = pids[int(i)]
        dev.add(p)
        n += sum(1 for it in items if it["problem_id"] == p and it["kind"] == "real")
    return dev


def build(cfg, rng):
    traces = {t["problem_id"]: t for t in read_jsonl(cfg.dir / "traces" / "traces.jsonl")
              if t.get("ok")}
    verbs = read_jsonl(cfg.dir / "nla" / "verbalisations.jsonl")
    pids = sorted(traces)
    items = []
    for kind, v, s in sample_items(verbs, rng):
        shown = v["problem_id"]
        if kind == "mismatched_trace":
            shown = pids[(pids.index(shown) + 1) % len(pids)]
        t = traces[shown]
        c = min(int(v["chunk_index"]), len(t["chunks"]) - 1)
        end = t["offsets"][t["chunks"][c]["token_end"]][1]
        items.append({"kind": kind, "problem_id": v["problem_id"], "shown_problem": shown,
                      "question": t["question"], "context": t["trace"][:end],
                      "continuation": t["trace"][end:end + 200],
                      "claims": [c.text for c in split_claims(s["explanation"])]})
    dev = split_by_problem(items, rng)
    order = rng.permutation(len(items))
    rows = {"dev": [], "heldout": []}
    key = {}
    for n, i in enumerate(order):
        it = items[int(i)]
        part = "dev" if it["problem_id"] in dev else "heldout"
        item_id = f"item{n:03d}"
        key[item_id] = {"kind": it["kind"], "problem_id": it["problem_id"],
                        "shown_problem": it["shown_problem"], "part": part}
        for j, claim in enumerate(it["claims"]):
            rows[part].append({"item": item_id, "claim": j, "question": it["question"],
                               "context_end": it["context"][-600:],
                               "continuation": it["continuation"], "claim_text": claim,
                               **{f: "" for f in FIELDS}, "note": ""})
    return rows, key


def score(path_a: str, path_b: str) -> dict:
    def load(p):
        with open(p, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    a, b = load(path_a), load(path_b)
    if [(r["item"], r["claim"]) for r in a] != [(r["item"], r["claim"]) for r in b]:
        raise SystemExit("the two sheets do not list the same claims in the same order")
    for rows, p in ((a, path_a), (b, path_b)):
        blank = sum(1 for r in rows for f in FIELDS if not r[f].strip())
        if blank:
            raise SystemExit(f"{p} has {blank} blank labels; nothing is scored from it")
    return {f: cohens_kappa([r[f] for r in a], [r[f] for r in b]) for f in FIELDS}


def report(path_a: str, path_b: str, key: dict, reliable_at: float = 0.6) -> dict:
    """What the D71 protocol's Part B reports, per field (D79 addition).

    ``score`` pools real claims with the blind controls, whose labels are easy
    to agree on, and gives no interval. This keeps them apart:

    * kappa on **held-out real** claims only, with a problem-cluster bootstrap
      interval, ``undefined`` when one label is used throughout;
    * each rater's label shares for real claims and for each kind of blind
      control, so a scheme that calls controls "supported" is visible;
    * ``reliable`` iff kappa is defined and at least ``reliable_at``. Rates of
      an unreliable field are listed but must not be interpreted.
    """
    from nlaast.faithfulness.claim_checks import kappa_summary

    score(path_a, path_b)                       # same rows, no blank labels
    def load(p):
        with open(p, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    a, b = load(path_a), load(path_b)
    items = key["items"]
    missing = sorted({r["item"] for r in a} - set(items))
    if missing:
        raise SystemExit(f"items not in the key: {missing[:5]}")
    held = [i for i, r in enumerate(a) if items[r["item"]]["part"] == "heldout"]
    if not held:
        raise SystemExit("no held-out items in these sheets; the protocol scores held-out items only")
    real = [i for i in held if items[a[i]["item"]]["kind"] == "real"]
    out: dict = {"n_heldout_claims": len(held), "n_heldout_real_claims": len(real), "fields": {}}
    for f in FIELDS:
        ks = kappa_summary([a[i][f].strip() for i in real], [b[i][f].strip() for i in real],
                           [items[a[i]["item"]]["problem_id"] for i in real])
        shares: dict = {}
        for kind in ("real", "gaussian_control", "mismatched_trace"):
            idx = [i for i in held if items[a[i]["item"]]["kind"] == kind]
            for name, rows in (("rater_a", a), ("rater_b", b)):
                labs = [rows[i][f].strip() for i in idx]
                shares.setdefault(kind, {})[name] = (
                    {lab: labs.count(lab) / len(labs) for lab in sorted(set(labs))} if labs else None)
        out["fields"][f] = {
            "kappa_heldout_real": ks,
            "reliable": bool(ks["defined"] and ks["kappa"] >= reliable_at),
            "kappa_pooled_with_controls": cohens_kappa([a[i][f].strip() for i in held],
                                                       [b[i][f].strip() for i in held]),
            "label_shares_heldout": shares,
        }
    return out


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--score", nargs=2, metavar=("A_CSV", "B_CSV"))
    p.add_argument("--report", nargs=2, metavar=("A_CSV", "B_CSV"),
                   help="the protocol's per-field report: held-out real claims, cluster "
                        "CI, control label shares (needs the run's key.json)")
    args = p.parse_args()
    if args.score:
        print(score(*args.score))
        return 0
    if args.report:
        import json

        cfg = load_config(args)
        key = json.loads((cfg.dir.parent / f"{cfg.run_id}_claim_annotation" / "key.json")
                         .read_text(encoding="utf-8"))
        print(json.dumps(report(*args.report, key), indent=1))
        return 0
    cfg = load_config(args)
    out = cfg.dir.parent / f"{cfg.run_id}_claim_annotation"
    if (out / "key.json").exists() and not args.force:
        print(f"{out} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    rows, key = build(cfg, np.random.default_rng(SEED))
    out.mkdir(parents=True, exist_ok=True)
    for part, rs in rows.items():
        with open(out / f"{part}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    write_json(out / "key.json", {"seed": SEED, "status": "blank; not annotated", "items": key})
    print({part: len(rs) for part, rs in rows.items()}, "claims; key:", len(key), "items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
