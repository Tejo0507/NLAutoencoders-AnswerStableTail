"""Stage `data`: benchmark acquisition, validation and deterministic splitting.

Downloads GSM8K and the configured MATH subjects if they are not already staged
under ``data/raw/``, validates every record parses to a gold answer, and emits
the study's problem set with its train/eval assignment.

The split is by hash of the problem id, so it is stable as the problem set
grows: a probe fitted during the pilot is still evaluated on held-out problems
in the main run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _stage import base_parser, paths, setup, should_skip

from nlaast.data import datasets
from nlaast.data.answers import PERMISSIVE, STRICT, equivalent, extract_answer
from nlaast.logging_utils import write_json, write_jsonl

STAGE = "data"

#: HuggingFace sources, used only when the local JSONL is absent.
HF_SOURCES = {
    "gsm8k": ("openai/gsm8k", "main"),
    "math": ("EleutherAI/hendrycks_math", None),
}


def ensure_gsm8k(split: str, log) -> Path:
    out = paths.DATA_RAW / "gsm8k" / f"{split}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        log.info("gsm8k/%s already staged (%d lines)", split,
                 sum(1 for _ in open(out, encoding="utf-8")))
        return out
    from datasets import load_dataset

    log.info("downloading GSM8K %s", split)
    ds = load_dataset(*HF_SOURCES["gsm8k"], split=split)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in ds:
            fh.write(json.dumps({"question": r["question"], "answer": r["answer"]},
                                ensure_ascii=False) + "\n")
    return out


def ensure_math(subject: str, split: str, log) -> Path:
    out = paths.DATA_RAW / "math" / split / f"{subject}.jsonl"
    if out.exists() and out.stat().st_size > 0:
        log.info("math/%s/%s already staged", split, subject)
        return out
    from datasets import load_dataset

    log.info("downloading MATH %s/%s", split, subject)
    ds = load_dataset(HF_SOURCES["math"][0], subject, split=split)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in ds:
            fh.write(json.dumps(
                {"problem": r["problem"], "level": r.get("level"),
                 "type": r.get("type", subject), "solution": r["solution"]},
                ensure_ascii=False) + "\n")
    return out


def parser_agreement(problems) -> dict:
    """How often the strict and permissive extractors agree on the gold answer.

    This is the baseline for falsification test F4. If the two parsers already
    disagree on *gold* solutions, any disagreement they show on generated traces
    is partly a parser artefact and the F4 result has to be read with that in
    mind.
    """
    n = agree = strict_null = 0
    for p in problems:
        if not p.gold_solution:
            continue
        n += 1
        a = extract_answer(p.gold_solution, PERMISSIVE)
        b = extract_answer(p.gold_solution, STRICT)
        if b is None:
            strict_null += 1
        if equivalent(a, b):
            agree += 1
    return {
        "n_checked": n,
        "agreement_rate": agree / n if n else float("nan"),
        "strict_unparseable_rate": strict_null / n if n else float("nan"),
    }


def gold_recovery(problems) -> dict:
    """Can the permissive extractor recover the gold answer from the gold solution?

    An upper bound on answer-extraction accuracy. If it is low, every downstream
    "the answer changed" measurement inherits that error rate, and the execution
    report has to say so.
    """
    hits = sum(
        1 for p in problems
        if p.gold_solution and equivalent(extract_answer(p.gold_solution, PERMISSIVE), p.gold)
    )
    n = sum(1 for p in problems if p.gold_solution)
    return {"n_checked": n, "recovery_rate": hits / n if n else float("nan")}


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0
    manifest.start_stage(STAGE)

    dcfg = cfg.data
    staged = {}
    if "gsm8k" in dcfg.datasets:
        staged["gsm8k"] = str(ensure_gsm8k(dcfg.split, log))
    if "math" in dcfg.datasets:
        for subj in dcfg.math_subjects:
            staged[f"math/{subj}"] = str(ensure_math(subj, dcfg.split, log))

    problems = datasets.load_problems(dcfg)
    if args.limit:
        problems = problems[: args.limit]
    report = datasets.validate(problems)
    report["parser_agreement"] = parser_agreement(problems)
    report["gold_recovery"] = gold_recovery(problems)
    report["sources"] = staged

    out_dir = cfg.stage_dir(STAGE)
    write_jsonl(out_dir / "problems.jsonl",
                              [{"id": p.id, **p.to_dict()} for p in problems])
    write_json(out_dir / "validation.json", report)

    log.info("problems: %d (%s)", report["n"], report["by_dataset"])
    log.info("split: %s", report["by_split"])
    log.info("gold recovery by the permissive parser: %.3f",
             report["gold_recovery"]["recovery_rate"])
    log.info("strict/permissive agreement on gold solutions: %.3f",
             report["parser_agreement"]["agreement_rate"])

    if not report["ok"]:
        manifest.finish_stage(STAGE, status="failed", metrics=report)
        log.error("validation failed: %s", report)
        return 1

    manifest.record_resource("datasets", staged=staged, split=dcfg.split,
                             n_problems=report["n"],
                             by_dataset=report["by_dataset"])
    manifest.finish_stage(STAGE, status="complete",
                          output=str(out_dir / "problems.jsonl"), metrics=report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
