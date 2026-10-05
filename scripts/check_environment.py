"""Stage `env`: environment and hardware report.

Runs first and writes what the rest of the pipeline is actually running on. The
hardware here is tight enough (6.44 GB VRAM against three 7B models) that the
feasibility checks below are part of the experiment record, not a convenience.
"""

from __future__ import annotations

import shutil
import sys

from _stage import base_parser, paths, provenance, setup, should_skip

from nlaast.logging_utils import write_json

STAGE = "env"


def disk_report() -> dict:
    out = {}
    for label, p in (("repo", paths.ROOT), ("model_cache", paths.model_cache()),
                     ("quantised", paths.quantised_models())):
        try:
            p.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(p)
            out[label] = {
                "path": str(p),
                "total_gb": round(usage.total / 1e9, 1),
                "free_gb": round(usage.free / 1e9, 1),
            }
        except Exception as exc:
            out[label] = {"path": str(p), "error": repr(exc)}
    return out


def feasibility(env: dict, disks: dict) -> dict:
    """Can this machine run the stages that need a GPU?"""
    checks = []

    vram = env.get("gpu_vram_gb")
    checks.append({
        "check": "cuda_available",
        "ok": bool(env.get("cuda_available")),
        "detail": f"torch {env.get('torch')} / cuda {env.get('cuda_version')}",
    })
    checks.append({
        "check": "vram_for_nf4_7b",
        "ok": bool(vram and vram >= 5.5),
        "detail": f"{vram} GB present; a 4-bit Qwen-7B needs roughly 5.0-5.5 GB "
                  f"with lm_head left in bf16",
    })
    free = disks.get("quantised", {}).get("free_gb", 0)
    checks.append({
        "check": "disk_for_quantised_models",
        "ok": bool(free and free >= 15),
        "detail": f"{free} GB free where NF4 checkpoints are written; the three "
                  f"models need about 15 GB once converted",
    })
    stage_free = disks.get("model_cache", {}).get("free_gb", 0)
    checks.append({
        "check": "disk_for_bf16_staging",
        "ok": bool(stage_free and stage_free >= 16),
        "detail": f"{stage_free} GB free in the download cache; the largest "
                  f"single bf16 checkpoint is 15.3 GB and is evicted after "
                  f"conversion",
    })
    ram = env.get("ram_total_gb")
    checks.append({
        "check": "ram",
        "ok": bool(ram and ram >= 8),
        "detail": f"{ram} GB total, {env.get('ram_available_gb')} GB free; "
                  f"conversion streams one shard at a time",
    })
    try:
        import bitsandbytes  # noqa: F401
        bnb_ok, bnb_detail = True, "importable"
    except Exception as exc:
        bnb_ok, bnb_detail = False, repr(exc)
    checks.append({"check": "bitsandbytes", "ok": bnb_ok, "detail": bnb_detail})

    try:
        from math_verify import parse, verify
        ok = bool(verify(parse("$1/2$", parsing_timeout=None),
                         parse("$0.5$", parsing_timeout=None), timeout_seconds=None))
        mv_detail = f"symbolic equivalence 1/2 == 0.5 -> {ok}"
    except Exception as exc:
        ok, mv_detail = False, repr(exc)
    checks.append({"check": "math_verify", "ok": ok, "detail": mv_detail})

    return {
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks),
        "blocking": [c["check"] for c in checks if not c["ok"]],
    }


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0

    manifest.start_stage(STAGE)
    env = provenance.environment()
    disks = disk_report()
    feas = feasibility(env, disks)

    report = {"environment": env, "disks": disks, "feasibility": feas,
              "git": provenance.git_commit()}
    out = cfg.stage_dir(STAGE) / "environment.json"
    write_json(out, report)

    for c in feas["checks"]:
        log.info("%-28s %s  %s", c["check"], "OK  " if c["ok"] else "FAIL", c["detail"])
    if not feas["all_ok"]:
        log.warning("failing checks: %s", ", ".join(feas["blocking"]))

    manifest.finish_stage(
        STAGE,
        status="complete",
        output=str(out),
        metrics={"all_checks_ok": feas["all_ok"], "blocking": feas["blocking"]},
    )
    # A failing check is information, not a reason to abort the pipeline; the
    # orchestrator decides what to do with it.
    return 0


if __name__ == "__main__":
    sys.exit(main())
