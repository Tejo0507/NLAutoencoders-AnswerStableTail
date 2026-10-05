"""Top-level orchestrator.

Runs the stage DAG in dependency order, skipping stages the manifest already
records as complete, and stops on the first hard failure unless told not to.

    python scripts/run_pipeline.py --config pilot
    python scripts/run_pipeline.py --config main --from traces
    python scripts/run_pipeline.py --config main --stages ast baselines analysis
    python scripts/run_pipeline.py --config main --dry-run

**The model-eviction ordering is not cosmetic.** Three Qwen-7B-shaped
checkpoints total 41 GB in bf16 against ~22 GB of staging disk, so only one can
be present at a time. Falsification test F8 compares 4-bit activations against
bf16 ones and therefore *must* run while the bf16 target is still on disk -
after the verbaliser is fetched, re-measuring drift costs a fresh 15 GB
download. The orchestrator inserts ``robustness:F8`` between ``acts`` and
``nla`` for exactly that reason, and ``--no-f8-ordering`` disables it.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlaast import config as config_mod  # noqa: E402
from nlaast import logging_utils, provenance  # noqa: E402

#: stage -> (script, dependencies)
STAGES: dict[str, tuple[str, tuple[str, ...]]] = {
    "env":          ("check_environment.py", ()),
    "data":         ("prepare_problems.py", ()),
    "models":       ("prepare_checkpoints.py", ()),
    "traces":       ("generate_traces.py", ("data", "models")),
    "acts":         ("extract_activations.py", ("traces",)),
    # Deliberately independent of `acts`: the tail must be defined without
    # reference to any activation (limitation L1 / objective O1). The DAG is
    # where that independence is enforced.
    "ast":          ("detect_answer_stable_tail.py", ("traces",)),
    "baselines":    ("score_baselines.py", ("acts", "ast")),
    "nla":          ("run_autoencoder.py", ("acts", "ast")),
    "faithfulness": ("audit_faithfulness.py", ("nla",)),
    "causal":       ("run_interventions.py", ("traces", "ast")),
    "analysis":     ("aggregate_results.py", ("ast", "baselines")),
    "robustness":   ("run_falsification_tests.py", ("ast", "acts")),
    "report":       ("write_run_report.py", ("analysis",)),
}

#: Extra arguments a stage needs when run from the orchestrator.
STAGE_ARGS: dict[str, list[str]] = {
    "ast": ["--sweep"],              # F5 needs the sweep and it is nearly free
    "acts": ["--layers", "14", "20", "24"],  # F7 needs the extra layers
}


def topological(selected: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dep in STAGES[name][1]:
            if dep in selected:
                visit(dep)
        out.append(name)

    for s in selected:
        visit(s)
    return out


def run_stage(stage: str, script: str, cfg_args: list[str], extra: list[str],
              log) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *cfg_args, *extra]
    log.info("=" * 72)
    log.info("STAGE %s  ->  %s", stage.upper(), " ".join(cmd[1:]))
    log.info("=" * 72)
    t0 = time.monotonic()
    rc = subprocess.run(cmd, cwd=str(ROOT / "scripts")).returncode
    log.info("stage %s finished rc=%d in %.1f min", stage, rc,
             (time.monotonic() - t0) / 60)
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="pilot")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("--run-id", default=None)
    p.add_argument("--stages", nargs="*", default=None,
                   help="stages to run (default: the config's stage list)")
    p.add_argument("--from", dest="start_from", default=None,
                   help="start at this stage and run everything after it")
    p.add_argument("--force", action="store_true",
                   help="redo stages the manifest records as complete")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-f8-ordering", action="store_true",
                   help="do not insert the early F8 run before the NLA stage")
    args = p.parse_args()

    cfg = config_mod.load(args.config, args.overrides)
    if args.run_id:
        cfg = config_mod.from_mapping({**cfg.to_dict(), "run_id": args.run_id})
    cfg.dir.mkdir(parents=True, exist_ok=True)
    log = logging_utils.setup(cfg.dir, name="orchestrator")
    manifest = provenance.Manifest.open(cfg.dir, cfg.to_dict(), cfg.hash())

    selected = list(args.stages) if args.stages else list(cfg.stages)
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        log.error("unknown stages: %s (known: %s)", unknown, sorted(STAGES))
        return 2
    if args.start_from:
        if args.start_from not in selected:
            log.error("--from %r is not in the selected stages", args.start_from)
            return 2
        selected = selected[selected.index(args.start_from):]

    order = topological(selected)
    cfg_args = ["--config", args.config]
    for o in args.overrides:
        cfg_args += ["--set", o]
    if args.run_id:
        cfg_args += ["--run-id", args.run_id]
    if args.force:
        cfg_args.append("--force")

    plan: list[tuple[str, str, list[str]]] = []
    for s in order:
        plan.append((s, STAGES[s][0], STAGE_ARGS.get(s, [])))
        # F8 before the verbaliser download evicts the bf16 target. See the
        # module docstring - this is a hard ordering constraint, not a
        # preference.
        if (s == "acts" and "nla" in order and not args.no_f8_ordering
                and "F8_quantisation" in cfg.robustness.enabled_tests):
            plan.append(("robustness:F8", "run_falsification_tests.py",
                         ["--only", "F8_quantisation", "--force"]))

    log.info("run=%s  config_hash=%s", cfg.run_id, cfg.hash())
    log.info("plan: %s", " -> ".join(s for s, _, _ in plan))
    if args.dry_run:
        for s, script, extra in plan:
            status = manifest.stage_status(s.split(":")[0])
            log.info("  %-16s %-30s %-8s %s", s, script, status, " ".join(extra))
        return 0

    failures = []
    started = time.monotonic()
    for stage, script, extra in plan:
        # A sub-step such as `robustness:F8` always runs - its own bookkeeping
        # happens inside the script, and the manifest entry belongs to the full
        # `robustness` stage, which has not run yet.
        is_substep = ":" in stage
        if not is_substep and manifest.is_complete(stage) and not args.force:
            log.info("skipping %s (already complete)", stage)
            continue
        rc = run_stage(stage, script, cfg_args, extra, log)
        if rc != 0:
            failures.append(stage)
            log.error("stage %s failed with rc=%d", stage, rc)
            if not args.continue_on_error:
                log.error("stopping. Re-run with --continue-on-error to push past it.")
                break

    elapsed = (time.monotonic() - started) / 60
    manifest = provenance.Manifest.open(cfg.dir)
    log.info("=" * 72)
    log.info("pipeline finished in %.1f min; failures: %s", elapsed, failures or "none")
    for s in cfg.stages:
        log.info("  %-14s %s", s, manifest.stage_status(s))
    log.info("outputs: %s", cfg.dir)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
