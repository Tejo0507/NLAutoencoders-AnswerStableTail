"""Stage `data` for a confirmatory run: problems no earlier run has seen (D75).

The same loaders, validation, train/eval split and stratified sampling as
``prepare_problems.py``, with one addition: every problem id appearing in any
existing ``runs/*/data/problems.jsonl`` (other than this run's own) is excluded
first. The selection is the next problems in the registered hash order after
the exclusions, so it is fixed by the data and the seed, not chosen.

    python scripts/prepare_fresh_problems.py --config replication
"""

from __future__ import annotations

import json
import sys

from _stage import base_parser, paths, setup, should_skip

from nlaast.data import datasets
from nlaast.logging_utils import read_jsonl, write_json, write_jsonl

STAGE = "data"


def used_ids(run_dir) -> dict[str, list[str]]:
    """Problem ids used by every other run, by run."""
    out = {}
    for f in sorted(paths.RUNS.glob("*/data/problems.jsonl")):
        if f.parent.parent == run_dir:
            continue
        out[f.parent.parent.name] = [r["id"] for r in read_jsonl(f)]
    return out


def select_fresh(dcfg, excluded: set[str]) -> list:
    groups = []
    if "gsm8k" in dcfg.datasets:
        pool = [p for p in datasets.load_gsm8k(dcfg) if p.id not in excluded]
        groups.append(datasets._take_stratified(pool, dcfg.n_gsm8k, dcfg.seed))
    if "math" in dcfg.datasets:
        pool = [p for p in datasets.load_math(dcfg) if p.id not in excluded]
        groups.append(datasets._take_stratified(pool, dcfg.n_math, dcfg.seed))
    return datasets.interleave_by_dataset(groups)


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0
    manifest.start_stage(STAGE)
    by_run = used_ids(cfg.dir)
    excluded = {i for ids in by_run.values() for i in ids}
    problems = select_fresh(cfg.data, excluded)
    overlap = sorted({p.id for p in problems} & excluded)
    if overlap:
        log.error("fresh selection overlaps earlier runs: %s", overlap[:10])
        manifest.finish_stage(STAGE, status="failed", metrics={"overlap": len(overlap)})
        return 1
    report = datasets.validate(problems)
    report["excluded_ids"] = len(excluded)
    report["excluded_from_runs"] = {k: len(v) for k, v in by_run.items()}
    out_dir = cfg.stage_dir(STAGE)
    write_jsonl(out_dir / "problems.jsonl", [{"id": p.id, **p.to_dict()} for p in problems])
    write_json(out_dir / "validation.json", report)
    log.info("fresh problems: %d (%s), %d ids excluded", report["n"],
             report["by_dataset"], len(excluded))
    manifest.finish_stage(STAGE, status="complete" if report["ok"] else "failed",
                          output=str(out_dir / "problems.jsonl"), metrics=report)
    print(json.dumps({k: report[k] for k in ("n", "by_dataset", "by_split", "excluded_ids")}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
