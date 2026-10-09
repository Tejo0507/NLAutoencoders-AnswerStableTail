"""The `models` stage's provenance record survives being run in two passes.

Three Qwen-7B-shaped checkpoints total 41 GB in bf16 against ~22 GB of staging
disk, so they cannot be prepared in one pass: the target goes first, F8
measures the quantisation drift against its bf16 weights, those weights are
evicted, and only then can the two autoencoders be fetched. Two passes over
this stage is the normal path, not an exception.

`resolved.json` was written from the current call's resolved set, so the second
pass - run with `--skip target` - replaced the file and dropped the target's
repo id, revision and local path. That record is the answer to "which weights
produced this result", and losing it is not recoverable from anything else in
the run directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_checkpoints import merge_resolved  # noqa: E402

TARGET = {"repo_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a354",
          "local_path": "D:/nf4/target", "precision": "nf4"}
AV = {"repo_id": "kitft/nla-qwen2.5-7b-L20-av", "revision": "b884691",
      "local_path": "D:/nf4/av", "precision": "nf4"}
AR = {"repo_id": "kitft/nla-qwen2.5-7b-L20-ar", "revision": "e2c9e57",
      "local_path": "E:/nf4/ar", "precision": "nf4"}


def test_the_first_pass_writes_what_it_prepared(tmp_path):
    out = merge_resolved(tmp_path / "resolved.json", {"target": TARGET})
    assert out == {"target": TARGET}


def test_the_second_pass_keeps_the_first_pass_record(tmp_path):
    """The eviction flow: target, then the autoencoders, in separate calls."""
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps({"target": TARGET}), encoding="utf-8")
    out = merge_resolved(path, {"av": AV, "ar": AR})
    assert sorted(out) == ["ar", "av", "target"]
    assert out["target"] == TARGET


def test_a_re_prepared_checkpoint_is_updated_not_duplicated(tmp_path):
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps({"target": TARGET}), encoding="utf-8")
    newer = {**TARGET, "revision": "newer-sha"}
    out = merge_resolved(path, {"target": newer})
    assert out == {"target": newer}


def test_a_missing_file_is_not_an_error(tmp_path):
    assert merge_resolved(tmp_path / "nope.json", {"av": AV}) == {"av": AV}
