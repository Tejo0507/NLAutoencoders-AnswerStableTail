"""Stage `models`: acquire and prepare the three checkpoints.

Target model, activation verbaliser, activation reconstructor. All three are
Qwen-7B-shaped and total 41 GB in bf16; this machine has about 22 GB of staging
disk and 6.44 GB of VRAM, so each is downloaded, converted to 4-bit NF4, saved,
and its bf16 source evicted before the next one starts (PROJECT_PLAN.md
section 2).

The stage also performs the compatibility checks that nothing downstream would
catch. The autoencoders are bound to one model at one layer and a mismatch
produces confident nonsense rather than an error, so the AV and AR sidecars are
cross-checked against each other and against the target configuration, and the
live tokeniser is verified to reproduce the pinned injection-token neighbours.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from _stage import base_parser, paths, setup, should_skip

from nlaast.models import loading
from nlaast.logging_utils import read_json, write_json
from nlaast.nla import meta as nla_meta

STAGE = "models"


def disk_free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def merge_resolved(path: Path, prepared: dict) -> dict:
    """Add this call's checkpoint records to whatever earlier calls recorded.

    The three checkpoints cannot coexist on this machine, so this stage is
    *expected* to run more than once with ``--skip``: the target first, then
    the autoencoders after the target's bf16 source has been evicted.
    Rewriting the file from the current call's resolved set would drop the
    provenance of everything an earlier call prepared - and that record is the
    answer to "which weights produced this result".
    """
    existing = read_json(path) if path.exists() else {}
    return {**existing, **prepared}


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--vendor-only", action="store_true",
                   help="only check the vendored upstream module, download nothing")
    p.add_argument("--keep-bf16", action="store_true",
                   help="do not evict bf16 sources (needs far more disk)")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["target", "av", "ar"], help="models to leave alone")
    p.add_argument("--evict-bf16", nargs="*", default=[],
                   choices=["target", "av", "ar"],
                   help="delete these models' bf16 download cache and exit. "
                        "The NF4 conversions are untouched. Needed because the "
                        "three checkpoints do not fit on the cache drive "
                        "together, and F8 must run before the target's bf16 "
                        "copy is removed.")
    args = p.parse_args()
    cfg, manifest, log = setup(args)

    # Always verify the vendored upstream - every NLA convention comes from it.
    try:
        up = nla_meta.upstream()
        log.info("vendored nla_inference loaded: %s",
                 [n for n in ("normalize_activation", "inject_at_marked_positions",
                              "NLACritic", "resolve_embed_scale") if hasattr(up, n)])
    except Exception as exc:
        log.error("vendored upstream unusable: %r", exc)
        return 1
    if args.vendor_only:
        return 0

    if args.evict_bf16:
        repos = {"target": cfg.target.repo_id, "av": cfg.nla.av_repo,
                 "ar": cfg.nla.ar_repo}
        for name in args.evict_bf16:
            repo = repos[name]
            before = disk_free_gb(paths.model_cache())
            try:
                local = loading.snapshot(repo)
                loading._evict_snapshot(Path(local), repo)
            except Exception as exc:
                log.warning("could not evict %s (%s): %r", name, repo, exc)
                continue
            after = disk_free_gb(paths.model_cache())
            log.info("evicted bf16 for %s: cache free %.1f -> %.1f GB",
                     name, before, after)
            manifest.record_resource(f"{name}_bf16_evicted", repo_id=repo,
                                     freed_gb=round(after - before, 1))
        return 0

    if should_skip(manifest, STAGE, args.force, log):
        return 0
    manifest.start_stage(STAGE)

    log.info("disk: cache=%.1f GB free, quantised=%.1f GB free",
             disk_free_gb(paths.model_cache()), disk_free_gb(paths.quantised_models()))

    wanted = [
        ("target", cfg.target.repo_id, cfg.target.revision),
        ("av", cfg.nla.av_repo, None),
        ("ar", cfg.nla.ar_repo, None),
    ]
    resolved: dict[str, loading.ResolvedModel] = {}

    for name, repo, revision in wanted:
        if name in args.skip:
            log.info("skipping %s (%s) by request", name, repo)
            continue
        free = disk_free_gb(paths.quantised_models())
        log.info("=== %s: %s | %.1f GB free for output ===", name, repo, free)
        try:
            resolved[name] = loading.ensure_nf4(
                repo, revision, evict_source=not args.keep_bf16
            )
        except Exception as exc:
            log.exception("failed to prepare %s (%s)", name, repo)
            manifest.finish_stage(STAGE, status="failed",
                                  error=f"{name}: {exc!r}")
            return 1
        manifest.record_resource(name, **resolved[name].provenance())
        log.info("%s ready at %s", name, resolved[name].local_path)

    # -- compatibility ----------------------------------------------------

    checks: dict = {}
    if "av" in resolved and "ar" in resolved:
        av = nla_meta.load_meta(resolved["av"].local_path)
        ar = nla_meta.load_meta(resolved["ar"].local_path)
        checks["sidecars"] = nla_meta.check_consistency(
            av, ar, cfg.target.layer, cfg.target.d_model
        )
        if not checks["sidecars"]["ok"]:
            log.error("checkpoint compatibility failed: %s",
                      checks["sidecars"]["problems"])
            write_json(cfg.stage_dir(STAGE) / "checks.json", checks)
            manifest.finish_stage(STAGE, status="failed", metrics=checks)
            return 1
        log.info("AV/AR/target are mutually consistent: layer=%d d_model=%d "
                 "inj_scale=%s mse_scale=%.4f",
                 av.extraction_layer_index, av.d_model, av.injection_scale, ar.mse_scale)
        if checks["sidecars"].get("note"):
            log.warning(checks["sidecars"]["note"])

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(resolved["av"].local_path))
        checks["tokenizer"] = nla_meta.verify_tokenizer(tok, av)
        if not checks["tokenizer"]["ok"]:
            log.error("injection-site verification failed: %s",
                      checks["tokenizer"].get("error"))
            write_json(cfg.stage_dir(STAGE) / "checks.json", checks)
            manifest.finish_stage(STAGE, status="failed", metrics=checks)
            return 1
        log.info("injection site verified at token %d (neighbours %d/%d)",
                 checks["tokenizer"]["position"], checks["tokenizer"]["left"],
                 checks["tokenizer"]["right"])

    out = cfg.stage_dir(STAGE)
    write_json(out / "checks.json", checks)

    record_path = out / "resolved.json"
    before = set(read_json(record_path)) if record_path.exists() else set()
    merged = merge_resolved(record_path,
                            {k: v.provenance() for k, v in resolved.items()})
    write_json(record_path, merged)
    for name in sorted(before - set(resolved)):
        log.info("kept the earlier provenance record for %s (%s)",
                 name, merged[name].get("local_path"))
    manifest.finish_stage(
        STAGE, status="complete", output=str(out),
        metrics={"prepared": sorted(resolved), "all_recorded": sorted(merged),
                 "skipped": sorted(args.skip), "checks_ok": True},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
