"""Does a run's analysis agree with its own artifacts, and with this code?

Read-only. Nothing here writes to the run directory, loads a model or touches
the GPU, so it is safe to use while a run is still generating traces.

Three kinds of disagreement are possible between a finished run and the
statements made about it, and they need different checks:

1. **The analysis disagrees with the artifacts it was computed from.** Every
   headline number in `analysis/results.json` is a function of a stage's own
   output, so it can be recomputed and diffed. This catches a summary written
   from one corpus and a report written from another.

2. **The analysis disagrees with itself.** The Benjamini-Hochberg q-values are
   a function of the p-values stored beside them, the family counts are a
   function of the test list, and several invariants the project has argued for
   in `DECISIONS.md` are checkable - most importantly D38's, that a McNemar
   comparison with zero discordant problems carries `p = 1.0` rather than
   nothing. These are cheap and they are the ones a reader cannot verify by
   eye.

3. **The artifacts predate the code that produced them.** This is the failure
   this script was written for. The pilot's `analysis` stage ran at commit
   31a30e3 on 2026-10-11; the corrections of D38 and D41 landed in
   `stats.py` and `aggregate_results.py` the following day. Every number in
   `analysis/results.json` still *reproduces* from the artifacts, so check (1)
   passes - but the probe comparison's stored p-value is `nan` where the
   current code gives 1.0, which moves the BH family from five evaluable tests
   to six and every q-value with it. Nothing in the run says so. The manifest
   records the commit each stage ran under, so this is decidable: ask git
   whether the code a stage depends on has changed since.

Usage:

    python scripts/check_run_consistency.py --run runs/pilot
    python scripts/check_run_consistency.py --run runs/main --no-staleness

Exit code 0 when every check passes, 1 when any fails.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from nlaast.analysis.stats import (  # noqa: E402
    TestResult, benjamini_hochberg, summarise_tests)

#: Which source paths each stage's output depends on. Used only for the
#: staleness check, so it needs to name the code that decides the *numbers*,
#: not every file the stage imports. Over-naming makes the check noisy;
#: under-naming makes it miss a correction, which is the failure it exists to
#: catch, so it errs towards naming too much.
STAGE_CODE: dict[str, tuple[str, ...]] = {
    "ast": ("scripts/detect_answer_stable_tail.py", "src/nlaast/ast_detect/",
            "src/nlaast/data/answers.py", "src/nlaast/trace/chunking.py"),
    "baselines": ("scripts/score_baselines.py", "src/nlaast/baselines/"),
    "nla": ("scripts/run_autoencoder.py", "src/nlaast/nla/"),
    "faithfulness": ("scripts/audit_faithfulness.py",
                     "src/nlaast/faithfulness/"),
    "causal": ("scripts/run_interventions.py", "src/nlaast/causal/"),
    "analysis": ("scripts/aggregate_results.py", "src/nlaast/analysis/",
                 "src/nlaast/baselines/stopping.py"),
    "robustness": ("scripts/run_falsification_tests.py",),
    "report": ("scripts/write_run_report.py", "src/nlaast/analysis/tables.py",
               "src/nlaast/viz/"),
}

#: The causal stage writes two layers, made by different code at different
#: times. `interventions.jsonl` is the measurement, generated on the GPU by the
#: symbols below; `summary.json` and everything downstream are *derived* from
#: those rows on CPU and can be rebuilt (`run_interventions.py
#: --summarise-only`). Both layers live in the same two files, so a file-level
#: history check cannot separate them: a correction to the verdict made the
#: rows look stale (PW-23). The rows are judged by the source of the code that
#: generates them, compared symbol by symbol with docstrings and comments
#: ignored; the derived layer is judged by recomputing it.
CAUSAL_ROW_SYMBOLS: dict[str, tuple[str, ...]] = {
    "scripts/run_interventions.py": ("run_problem", "fit_tail_direction",
                                     "chunks_of"),
    "src/nlaast/causal/interventions.py": (
        "InterventionResult", "VERIFICATION_MARKERS", "count_verification_markers",
        "projection_editor", "random_direction", "filler_text",
        "SPAN_SLACK_TOKENS", "intervention_spans", "spans_are_disjoint"),
    "src/nlaast/baselines/probe.py": ("difference_in_means_direction",),
    "src/nlaast/data/datasets.py": ("stratified_cap",),
}
#: Generation dependencies with no derived code in them, checked whole.
CAUSAL_ROW_FILES: tuple[str, ...] = (
    "src/nlaast/models/target.py", "src/nlaast/data/answers.py",
    "src/nlaast/trace/chunking.py", "src/nlaast/seeding.py",
    "src/nlaast/activations/store.py")
#: The verdict fields a stale summary or analysis would get wrong.
VERDICT_KEYS = ("supported", "beats_random_direction", "beats_matched_position",
                "monotone_dose_response", "answers_preserved", "outcome_varies")

TOL = 1e-9


class Report:
    """Collects outcomes so one run of the script prints everything it found."""

    def __init__(self, verbose: bool = False):
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.checked = 0
        self.verbose = verbose

    def ok(self, label: str, detail: str = "") -> None:
        self.checked += 1
        if self.verbose:
            print(f"  ok       {label}{(' - ' + detail) if detail else ''}")

    def fail(self, label: str, detail: str) -> None:
        self.checked += 1
        self.failures.append(f"{label}: {detail}")
        print(f"  MISMATCH {label}\n             {detail}")

    def warn(self, label: str, detail: str) -> None:
        self.warnings.append(f"{label}: {detail}")
        print(f"  warning  {label}\n             {detail}")

    def skip(self, label: str, why: str) -> None:
        print(f"  skipped  {label} - {why}")

    def compare(self, label: str, got: Any, want: Any) -> bool:
        if _same(got, want):
            self.ok(label, f"{got!r}")
            return True
        self.fail(label, f"recomputed {got!r}, stored {want!r}")
        return False


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
        return abs(float(a) - float(b)) <= TOL * max(1.0, abs(float(b)))
    return a == b


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


# --- 1. the analysis against the artifacts it came from ---


def check_rq1(run: Path, results: dict, rep: Report) -> None:
    """Re-derive the whole `ast` summary from `ast.jsonl` and diff it.

    Uses the stage's own `summarise`, so this checks that the stored summary is
    the one this code produces from these rows - which is both the
    reproducibility claim and, when it fails, the staleness signal.
    """
    rows = read_jsonl(run / "ast" / "ast.jsonl")
    if not rows:
        return rep.skip("rq1", "no ast/ast.jsonl")
    from detect_answer_stable_tail import summarise

    fresh = summarise(rows)
    for target, name in ((run / "ast" / "summary.json", "ast/summary.json"),
                         (None, "results.json rq1")):
        if target is not None and not target.exists():
            rep.skip(name, "absent")
            continue
        stored = read_json(target) if target else (results.get("rq1") or {})
        if not stored:
            rep.skip(name, "absent")
            continue
        for key, want in sorted(stored.items()):
            if key not in fresh or isinstance(want, (dict, list, str)):
                continue
            rep.compare(f"{name}.{key}", fresh[key], want)
        # The nested blocks carry the numbers the report quotes, so they are
        # checked too rather than skipped as "not scalar".
        for block in ("determinacy_vs_statement", "forced_answer_budget",
                      "convergence_vs_ast", "status_counts", "status_shares"):
            if block not in stored or block not in fresh:
                continue
            for key, want in sorted(stored[block].items()):
                if isinstance(want, (dict, list, str)) or key not in fresh[block]:
                    continue
                rep.compare(f"{name}.{block}.{key}", fresh[block][key], want)


def check_rq2(run: Path, results: dict, rep: Report) -> None:
    """The claim-level headline against the per-claim audit rows."""
    audits = read_jsonl(run / "faithfulness" / "audits.jsonl")
    head = results.get("rq2_headline") or {}
    if not audits or not head:
        return rep.skip("rq2_headline", "no audits.jsonl or no headline")
    claims = [c for a in audits for c in a.get("claims", [])]
    n = len(claims)
    dep = sum(1 for c in claims if c["reconstruction_dependent"])
    noise = sum(1 for c in claims if c["reproduced_under_noise"])
    both = sum(1 for c in claims
               if c["reconstruction_dependent"] and not c["reproduced_under_noise"])
    rep.compare("rq2_headline.n_claims", n, head["n_claims"])
    rep.compare("rq2_headline.reconstruction_dependent", dep / n,
                head["reconstruction_dependent"])
    rep.compare("rq2_headline.reproduced_from_noise", noise / n,
                head["reproduced_from_noise"])
    rep.compare("rq2_headline.dependent_and_not_noise", both / n,
                head["dependent_and_not_noise"])
    # The headline must be the conjunction it claims to be, not the difference
    # of the two rates - those coincide only when no dependent claim is also
    # reproduced from noise.
    if both > dep - noise and abs(head["dependent_and_not_noise"]
                                  - (dep - noise) / n) < TOL:
        rep.warn("rq2_headline.dependent_and_not_noise",
                 "equals dependent minus noise rather than the conjunction")


def check_reconstruction(run: Path, results: dict, rep: Report) -> None:
    """Per-window-kind fidelity against the reconstruction rows."""
    recon = read_jsonl(run / "nla" / "reconstructions.jsonl")
    stored = results.get("reconstruction") or []
    if not recon or not stored:
        return rep.skip("reconstruction", "no reconstructions.jsonl or no table")
    by: dict[str, list[float]] = {}
    for r in recon:
        by.setdefault(r["window_kind"], []).append(float(r["cosine"]))
    for row in stored:
        kind = row["window_kind"]
        if kind not in by:
            rep.fail(f"reconstruction[{kind}]",
                     "reported but absent from reconstructions.jsonl")
            continue
        vals = np.asarray(by[kind], dtype=float)
        rep.compare(f"reconstruction[{kind}].n", len(vals), row["n"])
        rep.compare(f"reconstruction[{kind}].mean_cosine",
                    float(vals.mean()), row["mean_cosine"])
        if row.get("median_cosine") is not None:
            rep.compare(f"reconstruction[{kind}].median_cosine",
                        float(np.median(vals)), row["median_cosine"])


def check_nla_arm(run: Path, results: dict, rep: Report) -> None:
    """The stopping arm's problem count and its exclusions.

    The arm's score is a mean over the samples that survive the integrity gate
    and exclude the noise control, so the counts have to add up to the file.
    """
    info = results.get("nla_stopping") or {}
    recon = read_jsonl(run / "nla" / "reconstructions.jsonl")
    if not info.get("available") or not recon:
        return rep.skip("nla_stopping", "arm unavailable or no reconstructions")
    usable = {r["problem_id"] for r in recon
              if r["window_kind"] != "gaussian_control" and r.get("integrity_ok")}
    if info.get("problem_ids"):
        extra = set(info["problem_ids"]) - usable
        if extra:
            rep.fail("nla_stopping.problem_ids",
                     f"{len(extra)} scored problems have no usable sample: "
                     f"{sorted(extra)[:5]}")
        else:
            rep.ok("nla_stopping.problem_ids", "all have usable samples")
    ex = info.get("exclusions")
    if not ex:
        return rep.warn(
            "nla_stopping.exclusions",
            "the arm reports no exclusion counts, so what its score is a mean "
            "over cannot be read from the output (added after this run)")
    rep.compare("nla_stopping.exclusions.n_samples_total", len(recon),
                ex["n_samples_total"])
    rep.compare(
        "nla_stopping.exclusions add up",
        ex["n_samples_used"] + ex["n_samples_excluded_as_control"]
        + ex["n_samples_excluded_by_integrity_gate"],
        ex["n_samples_total"])


# --- 2. the analysis against itself ---


def check_test_family(results: dict, rep: Report) -> None:
    """q-values, family counts and the invariants DECISIONS.md argues for."""
    block = results.get("tests") or {}
    tests = block.get("tests") or []
    if not tests:
        return rep.skip("test family", "no tests in results.json")
    q = results.get("fdr_q", 0.05)

    fresh = benjamini_hochberg(
        [TestResult(name=t["name"], statistic=0.0,
                    p_value=float("nan") if t["p_value"] is None else t["p_value"],
                    effect=0.0, effect_name=t["effect_name"], n=t["n"],
                    notes=dict(t.get("notes") or {}))
         for t in tests], q)
    by_name = {t.name: t for t in fresh}
    for t in tests:
        f = by_name[t["name"]]
        label = f"q_value[{t['name'][:48]}]"
        if t["q_value"] is None and f.q_value is None:
            rep.ok(label, "unevaluable in both")
        else:
            rep.compare(label, f.q_value, t["q_value"])
        if t["significant"] is not None:
            rep.compare(f"significant[{t['name'][:44]}]",
                        f.significant, t["significant"])

    summary = summarise_tests(fresh, q)
    for key in ("n_tests", "n_evaluated", "n_significant"):
        if key in block:
            rep.compare(f"tests.{key}", summary[key], block[key])
    for key in ("bh_family_size", "n_planned_but_unevaluable",
                "q_values_at_planned_family_size"):
        if key not in block:
            rep.warn(
                f"tests.{key}",
                "absent, so the family size this run corrected over - and the "
                "D38 sensitivity at the planned family size - cannot be read "
                "from the output (added after this run)")

    # D38: zero discordant problems is the null holding exactly, which is a
    # tested null and must carry p = 1.0. A nan there drops a test from the
    # BH denominator *because* its null was true, which biases every other
    # q-value downwards.
    for t in tests:
        notes = t.get("notes") or {}
        if notes.get("test") != "mcnemar_exact":
            continue
        if notes.get("n_discordant") == 0 and t["p_value"] is None:
            rep.fail(
                f"D38[{t['name'][:44]}]",
                "zero discordant problems with no p-value: the exact "
                "conditional p-value is 1.0, and recording nothing drops this "
                "test from the BH denominator because its null held exactly")
        elif notes.get("n_discordant") == 0:
            rep.compare(f"D38[{t['name'][:44]}].p_value", 1.0, t["p_value"])

    # A test that produced a p-value must carry its uncertainty. The project's
    # first commitment is that no point estimate is reported without one.
    for t in tests:
        notes = t.get("notes") or {}
        p = t["p_value"]
        if p is None or not np.isfinite(p) or notes.get("test") == "mann_whitney_u":
            continue
        if t["interval"] is None:
            rep.warn(f"interval[{t['name'][:44]}]",
                     "a p-value with no interval beside it")

    # Significance must follow from the q-value and the threshold.
    for t in tests:
        if t["significant"] is None or t["q_value"] is None:
            continue
        rep.compare(f"significant follows q[{t['name'][:36]}]",
                    bool(t["q_value"] <= q), t["significant"])


def check_o3_accounting(results: dict, rep: Report) -> None:
    """The primary comparison's counts, and that its direction is stated."""
    tests = [t for t in ((results.get("tests") or {}).get("tests") or [])
             if t["name"].startswith("O3:")]
    if not tests:
        return rep.skip("O3 accounting", "no O3 tests")
    for t in tests:
        n = t.get("notes") or {}
        label = t["name"].split("minus ")[-1].split(" at ")[0]
        if n.get("test") != "mcnemar_exact":
            # An unevaluable comparison must say why, and must not look like a
            # null: no p-value, no q-value, a stated reason.
            if not n.get("skipped"):
                rep.fail(f"O3[{label}]",
                         "neither tested nor marked as unevaluable")
            elif t["q_value"] is not None:
                rep.fail(f"O3[{label}]",
                         "unevaluable but carries a q-value, so it entered the "
                         "BH denominator")
            else:
                rep.ok(f"O3[{label}]", "unevaluable, with a reason and no q")
            continue
        rep.compare(f"O3[{label}].n_discordant",
                    n["a_safe_b_not"] + n["b_safe_a_not"], n["n_discordant"])
        if "direction" not in n:
            rep.fail(f"O3[{label}].direction",
                     "no direction recorded, so a significant q cannot be read "
                     "as a win or a loss")
        else:
            rate_nla = n.get("safe_rate_nla")
            rate_other = n.get(f"safe_rate_{label}")
            if rate_nla is not None and rate_other is not None:
                expected = ("nla_readout better" if rate_nla > rate_other else
                            "nla_readout worse" if rate_nla < rate_other else
                            "tied")
                rep.compare(f"O3[{label}].direction", expected, n["direction"])
        shared = n.get("n_problems_shared")
        if shared is not None:
            rep.compare(f"O3[{label}].n_problems is the shared set",
                        shared, n["n_problems"])


def check_causal(run: Path, results: dict, rep: Report) -> None:
    """The causal summary against its rows, and the analysis against both.

    Three ways the derived layer can be wrong, each decided from content:
    the rows changed after the summary was built from them (the summary names
    their SHA-256); the summary is not what the current code derives from
    those rows; or the analysis and report carry a verdict other than the
    summary's. A summary rebuilt over unchanged rows passes all three.
    """
    rows_path = run / "causal" / "interventions.jsonl"
    summary_path = run / "causal" / "summary.json"
    if not rows_path.exists() or not summary_path.exists():
        return rep.skip("causal", "no causal rows or summary")
    import hashlib
    from nlaast.causal.interventions import (InterventionResult,
                                             direction_claim_supported)

    summary = read_json(summary_path)
    sha = hashlib.sha256(rows_path.read_bytes()).hexdigest()
    built_from = summary.get("rows_sha256")
    if built_from is None:
        rep.warn("causal.rows_sha256",
                 "summary.json does not name the rows it was built from, so a "
                 "change to interventions.jsonl since cannot be ruled out; "
                 "rebuild it with run_interventions.py --summarise-only")
    elif built_from != sha:
        rep.fail("causal.rows_sha256",
                 f"interventions.jsonl has changed since summary.json was built "
                 f"from it (built from {built_from[:12]}, now {sha[:12]}); the "
                 f"summary and everything derived from it are stale")
    else:
        rep.ok("causal.rows_sha256", sha[:12])

    rows = read_jsonl(rows_path)
    derived = direction_claim_supported(
        [InterventionResult(**{k: v for k, v in r.items() if k != "id"})
         for r in rows])
    stored = summary.get("verdict") or {}
    for k in VERDICT_KEYS:
        rep.compare(f"causal.summary.verdict.{k}", derived.get(k), stored.get(k))

    # Downstream layers are held to the summary they are derived from, so a
    # stale analysis cannot vouch for a report built on it.
    analysed = (results.get("rq3") or {}).get("verdict")
    if analysed is not None:
        for k in VERDICT_KEYS:
            rep.compare(f"analysis.rq3.verdict.{k}", stored.get(k), analysed.get(k))

    basis = results.get("rq3_family_test_basis")
    if basis is not None:
        import aggregate_results as agg
        again = agg.rq3_paired_contrast(rows, 0.95)
        for k in ("n_pairs", "beats", "intervention_demonstrated",
                  "candidate_rows_without_continuation", "readout_constant_zero"):
            rep.compare(f"analysis.rq3_family_test_basis.{k}",
                        again.get(k), basis.get(k))

    report = run / "report" / "RESULTS.md"
    if report.exists():
        text = report.read_text(encoding="utf-8")
        said = ("not tested" if "**Direction claim: not tested.**" in text else
                (m.group(1) if (m := re.search(
                    r"\*\*Direction claim supported: (\w+)\*\*", text)) else None))
        if said is not None:
            want = ("not tested" if stored.get("supported") is None
                    else str(stored.get("supported")))
            rep.compare("report.rq3.direction_claim", want, said)


# --- 3. the artifacts against the code that produced them ---


def _strip_docstrings(node: Any) -> Any:
    import ast
    for n in ast.walk(node):
        body = getattr(n, "body", None)
        if (isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                           ast.Module))
                and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            n.body = body[1:] or [ast.Pass()]
    return node


def _symbol_sources(source: str) -> dict[str, str]:
    """Top-level definitions, keyed by name, as ASTs without docstrings.

    Comments, docstrings and layout do not reach the AST, so a change to them
    is not a change to what the code does.
    """
    import ast
    out: dict[str, str] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        dumped = ast.dump(_strip_docstrings(node))
        for name in names:
            out[name] = dumped
    return out


def causal_row_code_changes(commit: str) -> list[str] | None:
    """Row-generating code that differs between ``commit`` and HEAD.

    None when git cannot answer. Names a symbol as ``path:name``, and a whole
    file as its path.
    """
    changed: list[str] = []
    for path, names in CAUSAL_ROW_SYMBOLS.items():
        old, new = _git("show", f"{commit}:{path}"), _git("show", f"HEAD:{path}")
        if old is None or new is None:
            return None
        before, after = _symbol_sources(old), _symbol_sources(new)
        changed += [f"{path}:{n}" for n in names if before.get(n) != after.get(n)]
    log = _git("log", "--oneline", f"{commit}..HEAD", "--", *CAUSAL_ROW_FILES)
    if log is None:
        return None
    if log:
        changed += [line for line in log.splitlines()]
    return changed


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def check_staleness(run: Path, rep: Report) -> None:
    """Has a stage's code changed since the stage last ran?

    The manifest records the commit per stage, so this is a question git can
    answer. A stage whose code has moved is not necessarily wrong - a comment
    or a log line changes nothing - so the commit subjects are printed and the
    judgement is left to the reader.

    **A dirty tree makes the comparison inconclusive rather than negative.** A
    stage that ran with uncommitted changes may have run with exactly the code
    that later became the commits below, in which case nothing is stale; or
    with older code, in which case everything downstream is. Git cannot tell
    the two apart, so that case is a warning. Only a stage that ran from a
    clean tree, with its code since changed, is a failure.

    The checks in part 2 are the ones that settle it either way: they are
    computed from the artifacts rather than inferred from history, which is why
    the pilot's `analysis` stage fails there as well as here.
    """
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        return rep.skip("staleness", "no manifest.json")
    if _git("rev-parse", "HEAD") is None:
        return rep.skip("staleness", "not a git checkout, or git unavailable")
    stages = (read_json(manifest_path).get("stages") or {})
    for stage, paths in STAGE_CODE.items():
        record = stages.get(stage)
        if not record or record.get("status") != "complete":
            continue
        commit = (record.get("git") or {}).get("commit")
        if not commit:
            rep.warn(f"staleness[{stage}]",
                     "the manifest records no commit for this stage")
            continue
        if _git("cat-file", "-e", f"{commit}^{{commit}}") is None:
            rep.warn(f"staleness[{stage}]",
                     f"commit {commit[:9]} is not in this checkout")
            continue
        if stage == "causal":
            # The rows only; the derived layer is checked by recomputation in
            # check_causal, which a code change to it cannot slip past.
            changed = causal_row_code_changes(commit)
            if changed is None:
                rep.warn("staleness[causal]", "git could not compare the code")
            elif not changed:
                rep.ok("staleness[causal]",
                       f"row-generating code unchanged since {commit[:9]}")
            else:
                detail = (f"rows generated at {commit[:9]}; the code that "
                          f"generates them has changed since, so the rows may "
                          f"not be what this code would now produce:\n"
                          + "\n".join(f"               {c}" for c in changed[:8]))
                if (record.get("git") or {}).get("dirty"):
                    rep.warn("staleness[causal]", detail + (
                        "\n               the tree was dirty when it ran - git "
                        "cannot tell whether it ran with this code"))
                else:
                    rep.fail("staleness[causal]", detail)
            continue
        log = _git("log", "--oneline", f"{commit}..HEAD", "--", *paths)
        if log is None:
            rep.warn(f"staleness[{stage}]", "git log failed")
            continue
        if not log:
            rep.ok(f"staleness[{stage}]", f"code unchanged since {commit[:9]}")
            continue
        lines = log.splitlines()
        dirty = bool((record.get("git") or {}).get("dirty"))
        detail = (f"ran at {commit[:9]}; {len(lines)} commit(s) have touched "
                  f"its code since, so its artifacts may not be what this "
                  f"code would now produce:\n"
                  + "\n".join(f"               {line}" for line in lines[:8]))
        if dirty:
            detail += ("\n               the tree was dirty when it ran, so it "
                       "may have run with the code that became those commits - "
                       "git cannot tell; re-run the stage or check part 2")
            rep.warn(f"staleness[{stage}]", detail)
        else:
            rep.fail(f"staleness[{stage}]", detail)


# --- the report's prose against the analysis ---

#: Numbers the execution report states in words, with where each comes from.
#: Only quantities the report prints to a fixed precision are listed: the point
#: is to catch a report built from a different `results.json`, not to re-parse
#: the whole document.
PROSE_CHECKS: tuple[tuple[str, str, str], ...] = (
    (r"claims audited: \*\*(\d+)\*\*", "rq2_headline.n_claims", "int"),
    (r"reconstruction-dependent: \*\*([\d.]+)\*\*",
     "rq2_headline.reconstruction_dependent", "3f"),
    (r"dependent \*\*and\*\* not reproduced from noise: \*\*([\d.]+)\*\*",
     "rq2_headline.dependent_and_not_noise", "3f"),
    (r"tail fraction: mean \*\*([\d.]+)\*\*", "rq1.mean_tail_fraction", "3f"),
)


def _dig(results: dict, dotted: str) -> Any:
    node: Any = results
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check_report_prose(run: Path, results: dict, rep: Report) -> None:
    path = run / "report" / "RESULTS.md"
    if not path.exists():
        return rep.skip("report prose", "no report/RESULTS.md")
    text = path.read_text(encoding="utf-8")
    for pattern, dotted, kind in PROSE_CHECKS:
        m = re.search(pattern, text)
        want = _dig(results, dotted)
        if m is None or want is None:
            rep.skip(f"prose[{dotted}]", "not stated in the report")
            continue
        if kind == "int":
            rep.compare(f"prose[{dotted}]", int(m.group(1)), int(want))
        else:
            rep.compare(f"prose[{dotted}]", m.group(1), f"{float(want):.3f}")

    # The verdict table is the headline, so each row is checked against the
    # test it claims to summarise.
    for row in re.finditer(
            r"^\| (\w+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| ([^|]+) \| \*\*([^*]+)\*\* \| ([\d.]+) \|$",
            text, re.MULTILINE):
        label, nla_rate, other_rate, n_problems, _disc, direction, q = row.groups()
        test = next((t for t in ((results.get("tests") or {}).get("tests") or [])
                     if t["name"].startswith("O3:")
                     and f"minus {label} " in t["name"]), None)
        if test is None:
            continue
        notes = test.get("notes") or {}
        rep.compare(f"verdict table[{label}].safe_rate_nla", nla_rate,
                    f"{float(notes['safe_rate_nla']):.3f}")
        rep.compare(f"verdict table[{label}].safe_rate_{label}", other_rate,
                    f"{float(notes[f'safe_rate_{label}']):.3f}")
        rep.compare(f"verdict table[{label}].n_problems", int(n_problems),
                    int(notes["n_problems"]))
        rep.compare(f"verdict table[{label}].direction", direction.strip(),
                    notes["direction"])
        if test["q_value"] is not None:
            rep.compare(f"verdict table[{label}].q", q,
                        f"{float(test['q_value']):.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/pilot", type=Path)
    ap.add_argument("--no-staleness", action="store_true",
                    help="skip the check that a stage's code has not moved "
                         "since the stage ran")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print the checks that pass as well as those that fail")
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else ROOT / args.run
    results_path = run / "analysis" / "results.json"
    if not results_path.exists():
        print(f"no analysis/results.json under {run} - nothing to check")
        return 1
    results = read_json(results_path)
    rep = Report(args.verbose)

    print(f"checking {run}\n")
    print("1. the analysis against the artifacts it came from")
    check_rq1(run, results, rep)
    check_rq2(run, results, rep)
    check_reconstruction(run, results, rep)
    check_nla_arm(run, results, rep)
    check_causal(run, results, rep)
    print("\n2. the analysis against itself")
    check_test_family(results, rep)
    check_o3_accounting(results, rep)
    print("\n3. the report's prose against the analysis")
    check_report_prose(run, results, rep)
    if args.no_staleness:
        print("\n4. the artifacts against the code that produced them - skipped")
    else:
        print("\n4. the artifacts against the code that produced them")
        check_staleness(run, rep)

    print(f"\n{rep.checked} checks, {len(rep.failures)} mismatched, "
          f"{len(rep.warnings)} warning(s)")
    if rep.failures:
        print("\nmismatches:")
        for f in rep.failures:
            print(f"  - {f.splitlines()[0]}")
    if not rep.failures and not rep.warnings:
        print("every check passed")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
