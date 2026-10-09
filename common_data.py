"""Corpus loading and text features for the root-level supervised scripts.

`classification.py` and `regression.py` fit models over the trace corpus the
pipeline produces. The corpus lives in one directory per run under `runs/`,
and more than one run can hold traces for the same problem: the pilot and the
larger `mlcorpus` run were deliberately given identical generation and AST
settings so their rows pool, and `runs/` also accumulates smoke runs and the
fixture run that `tests/test_end_to_end_tiny.py` writes.

Pooling all of that would be wrong in two different ways. Traces generated
under a different token cap, a different chunker or a different K are not
rows of the same corpus; and the fixture run's traces come from a randomly
initialised 64-dimensional model, so including them would mean fitting a model
on noise and reporting the result as a finding.

So which runs are poolable is decided by comparing the settings that determine
what a trace *is*, not by directory name, and the decision is written out
alongside every result. A run that disagrees on any of them is excluded with
its reason recorded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import yaml

RUNS = Path("runs")

#: Settings two runs must agree on for their traces to be rows of one corpus.
#: Each one changes the trace, its chunking or the boundary evidence the AST
#: criteria are applied to, so a disagreement makes the rows incomparable
#: rather than merely noisier.
POOLING_KEYS: tuple[tuple[str, ...], ...] = (
    ("target", "repo_id"),
    ("target", "precision"),
    ("target", "layer"),
    ("generation", "max_new_tokens"),
    ("generation", "temperature"),
    ("generation", "top_p"),
    ("generation", "canonical_greedy"),
    ("generation", "system_prompt"),
    ("generation", "force_answer_suffix"),
    ("generation", "force_answer_max_new_tokens"),
    ("chunk", "min_chunk_chars"),
    ("chunk", "max_chunks"),
    ("ast", "k_continuations"),
    ("ast", "continuation_temperature"),
    ("ast", "continuation_max_new_tokens"),
    ("ast", "require_unanimous"),
    ("ast", "min_boundary_fraction"),
    ("ast", "max_boundaries_evaluated"),
)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resolved_config(run_dir: Path) -> dict | None:
    p = run_dir / "config.resolved.yaml"
    if not p.exists():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _manifest(run_dir: Path) -> dict:
    p = run_dir / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def code_revision(run_dir: Path) -> str:
    """How the traces in this run were produced, as far as it can be known.

    Settings are not the whole story: several answer-parser and chunker bugs
    were fixed after the first pilot, and traces written before those fixes
    have different chunk boundaries baked into them. They are not rows of the
    same corpus as traces written after, however well their configs match, and
    no amount of reparsing repairs a chunk boundary.

    A clean commit identifies the code exactly. A dirty tree does not, so each
    dirty run gets a value unique to itself and will never pool automatically
    with another run - the safe direction, and overridable with an explicit
    ``--runs`` when a human has checked.
    """
    git = _manifest(run_dir).get("git") or {}
    commit = git.get("commit") or "unknown"
    if git.get("dirty"):
        return f"dirty:{run_dir.name}:{commit[:12]}"
    return f"clean:{commit}"


def pooling_signature(run_dir: Path) -> dict:
    """The facts about a run that decide whether its traces pool with another's."""
    cfg = _resolved_config(run_dir) or {}
    out: dict = {}
    for key in POOLING_KEYS:
        node: object = cfg
        for part in key:
            node = (node or {}).get(part) if isinstance(node, dict) else None
        out[".".join(key)] = node
    out["code_revision"] = code_revision(run_dir)
    return out


def _n_traces(run_dir: Path) -> int:
    p = run_dir / "traces" / "traces.jsonl"
    if not p.exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _traces_mtime(run_dir: Path) -> float:
    p = run_dir / "traces" / "traces.jsonl"
    return p.stat().st_mtime if p.exists() else 0.0


def select_runs(run_ids: list[str] | None = None) -> dict:
    """Decide which runs form the corpus, and say why.

    The reference is the **most recently written** trace file, not the largest:
    the newest traces are the ones produced by the current code, and a large
    older run is exactly the thing that should not be allowed to outvote them.
    Every other run joins only if its pooling signature matches, which is what
    keeps the fixture run, the smoke runs and pre-fix traces out without
    naming any of them.

    With ``run_ids`` the choice is the caller's, and a requested run that is
    not poolable is an error rather than a silent inclusion.
    """
    available = sorted(d.name for d in RUNS.iterdir()
                       if d.is_dir() and _n_traces(d) > 0) if RUNS.is_dir() else []
    if not available:
        raise SystemExit(
            "no run under runs/ has traces. Generate some first:\n"
            "  .venv/Scripts/python.exe scripts/generate_traces.py --config pilot"
        )

    if run_ids:
        missing = [r for r in run_ids if r not in available]
        if missing:
            raise SystemExit(
                f"runs with no traces: {missing}. available: {available}"
            )
        chosen = list(run_ids)
        reference = chosen[0]
    else:
        reference = max(available, key=lambda r: _traces_mtime(RUNS / r))
        chosen = [reference]

    ref_sig = pooling_signature(RUNS / reference)
    excluded: dict[str, object] = {}

    def differences(name: str) -> dict:
        sig = pooling_signature(RUNS / name)
        return {k: {"this_run": sig[k], "reference": ref_sig[k]}
                for k in ref_sig if sig[k] != ref_sig[k]}

    if run_ids:
        for name in chosen[1:]:
            diff = differences(name)
            if diff:
                raise SystemExit(
                    f"run {name!r} was requested but is not poolable with "
                    f"{reference!r}; it differs on {sorted(diff)}"
                )
    else:
        for name in available:
            if name == reference:
                continue
            if _resolved_config(RUNS / name) is None:
                excluded[name] = "no config.resolved.yaml"
                continue
            diff = differences(name)
            if diff:
                excluded[name] = {"differs_on": sorted(diff), "detail": diff}
            else:
                chosen.append(name)
        chosen.sort()

    return {
        "runs": chosen,
        "reference_run": reference,
        "pooling_signature": ref_sig,
        "excluded": excluded,
        "available": available,
        "explicit": bool(run_ids),
    }


def load_traces(runs: list[str]):
    """One row per problem id, earliest listed run wins on a collision."""
    seen = {}
    per_run = {}
    for name in runs:
        path = RUNS / name / "traces" / "traces.jsonl"
        kept = 0
        for r in read_jsonl(path):
            if r.get("ok", True) and r["id"] not in seen:
                seen[r["id"]] = r
                kept += 1
        per_run[name] = kept
    return list(seen.values()), per_run


def load_ast(runs: list[str]):
    seen = {}
    for name in runs:
        path = RUNS / name / "ast" / "ast.jsonl"
        if not path.exists():
            continue
        for r in read_jsonl(path):
            seen.setdefault(r["id"], r)
    return seen


def text_features(row):
    trace = row["trace"]
    question = row["question"]
    words = re.findall(r"[A-Za-z']+", trace)
    return {
        "q_words": len(question.split()),
        "q_numbers": len(re.findall(r"\d+(?:\.\d+)?", question)),
        "trace_chars": len(trace),
        "trace_words": len(words),
        "n_tokens": row.get("n_tokens", len(trace) // 4),
        "n_chunks": row.get("n_chunks", trace.count("\n\n") + 1),
        "n_equals": trace.count("="),
        "n_latex": trace.count("\\"),
        "n_numbers": len(re.findall(r"\d+(?:\.\d+)?", trace)),
        "n_lines": trace.count("\n") + 1,
        "cue_verify": len(re.findall(r"\b(check|verify|confirm|double-check|wait|actually)\b", trace, re.I)),
        "cue_conclude": len(re.findall(r"\b(therefore|thus|hence|so)\b", trace, re.I)),
        "cue_answer": len(re.findall(r"answer is", trace, re.I)),
        "is_math": int(row.get("dataset") == "math"),
        "truncated": int(bool(row.get("truncated", False))),
    }


def build_table(with_ast=False, run_ids=None):
    """The feature table, plus the provenance of every row in it.

    Returns ``(DataFrame, provenance)``. The provenance is written beside the
    results by both callers: a table of model scores means nothing without the
    corpus it was fitted on, and at these sample sizes the corpus size is the
    single most important thing a reader needs.
    """
    selection = select_runs(run_ids)
    rows, per_run = load_traces(selection["runs"])
    ast = load_ast(selection["runs"]) if with_ast else {}

    records = []
    dropped_no_ast = 0
    for row in rows:
        rec = {"id": row["id"], "dataset": row.get("dataset")}
        rec.update(text_features(row))
        rec["final_correct"] = int(bool(row.get("final_correct")))
        if with_ast:
            a = ast.get(row["id"])
            if a is None or a["status"] != "ok":
                dropped_no_ast += 1
                continue
            rec["tail_fraction"] = a["tail_fraction"]
        records.append(rec)

    df = pd.DataFrame(records)
    provenance = {
        **selection,
        "traces_per_run": per_run,
        "n_traces_loaded": len(rows),
        "n_rows": len(df),
        "requires_ast": bool(with_ast),
        "dropped_without_ok_tail": dropped_no_ast if with_ast else 0,
        "note": ("Rows come only from runs whose pooling signature matches the "
                 "reference run; see POOLING_KEYS in common_data.py."),
    }
    return df, provenance


def write_provenance(out_dir, provenance, extra=None):
    out = dict(provenance)
    if extra:
        out.update(extra)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "provenance.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
