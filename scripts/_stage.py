"""Shared entry-point plumbing for the stage scripts.

Puts ``src/`` on the path, parses the arguments every stage accepts, and opens
the run's manifest. Keeping it here means a stage script is just its stage.

``logging_utils``, ``paths`` and ``provenance`` are re-exported so a stage can
do its imports in one line - they only resolve after ``src/`` is on the path,
which happens below.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nlaast import config as config_mod  # noqa: E402
from nlaast import logging_utils, paths, provenance, seeding  # noqa: E402,F401


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="pilot",
                   help="config overlay name or path (default: pilot)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a config value; repeatable")
    p.add_argument("--run-id", default=None, help="override config.run_id")
    p.add_argument("--force", action="store_true",
                   help="redo this stage even if the manifest says it is complete")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most N units of work (smoke testing)")
    return p


def setup(args) -> tuple:
    """Resolve config, prepare the run directory, open the manifest."""
    cfg = config_mod.load(args.config, args.overrides)
    if args.run_id:
        cfg = config_mod.from_mapping({**cfg.to_dict(), "run_id": args.run_id})

    cfg.dir.mkdir(parents=True, exist_ok=True)
    log = logging_utils.setup(cfg.dir)
    manifest = provenance.Manifest.open(cfg.dir, cfg.to_dict(), cfg.hash())
    config_mod.save(cfg, cfg.dir / "config.resolved.yaml")
    seeding.seed_everything(cfg.seed)
    log.info("run=%s config_hash=%s dir=%s", cfg.run_id, cfg.hash(), cfg.dir)
    return cfg, manifest, log


def should_skip(manifest, stage: str, force: bool, log) -> bool:
    if manifest.is_complete(stage) and not force:
        log.info("stage %r already complete - use --force to redo", stage)
        return True
    return False
