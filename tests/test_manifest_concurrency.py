"""Stage processes that share a manifest must not erase each other's records.

On `main`, `detect_answer_stable_tail.py` and the two supervised analyses ran
while `extract_activations.py` held the GPU, as the post-trace checklist
allows. Each process loaded `manifest.json` once and wrote its whole in-memory
copy back, so `acts`, finishing last, erased the `ast`, `regression_tail` and
`classify_correct` records - and the run report then listed `ast` as "not run".
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

from nlaast.provenance import Manifest

SRC = Path(__file__).resolve().parents[1] / "src"


def _on_disk(run_dir: Path) -> dict:
    return json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))


class TestStaleCopiesDoNotEraseEachOther:
    def test_the_main_run_sequence_keeps_both_stages(self, tmp_path):
        """The exact interleaving: both open, one finishes, the other finishes."""
        a = Manifest.open(tmp_path, {"x": 1}, "h1")
        b = Manifest.open(tmp_path, {"x": 1}, "h1")
        b.start_stage("acts")
        a.start_stage("ast")
        a.finish_stage("ast", metrics={"n": 250})
        b.finish_stage("acts")          # b's copy predates "ast"

        stages = _on_disk(tmp_path)["stages"]
        assert stages["ast"]["status"] == "complete"
        assert stages["ast"]["metrics"] == {"n": 250}
        assert stages["acts"]["status"] == "complete"

    def test_a_writer_sees_the_other_stages_after_it_writes(self, tmp_path):
        a = Manifest.open(tmp_path, {"x": 1}, "h1")
        b = Manifest.open(tmp_path, {"x": 1}, "h1")
        a.start_stage("ast")
        b.start_stage("acts")
        assert b.stage_status("ast") == "running"

    def test_resources_are_merged_too(self, tmp_path):
        a = Manifest.open(tmp_path, {"x": 1}, "h1")
        b = Manifest.open(tmp_path, {"x": 1}, "h1")
        a.record_resource("datasets", n=250)
        b.record_resource("target_model", repo="q")
        res = _on_disk(tmp_path)["resources"]
        assert set(res) >= {"datasets", "target_model"}

    def test_a_process_still_owns_its_own_stage(self, tmp_path):
        """Merging must not let a stale copy revert a stage it did not touch,
        nor stop it updating the one it did."""
        a = Manifest.open(tmp_path, {"x": 1}, "h1")
        a.start_stage("ast")
        b = Manifest.open(tmp_path, {"x": 1}, "h1")
        a.finish_stage("ast")
        b.start_stage("acts")           # b last saw "ast" as running
        assert _on_disk(tmp_path)["stages"]["ast"]["status"] == "complete"


WORKER = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    from pathlib import Path
    from nlaast.provenance import Manifest
    run_dir, tag, n = Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    m = Manifest.open(run_dir, {"x": 1}, "h1")
    for i in range(n):
        m.start_stage(f"{tag}_{i}")
        m.finish_stage(f"{tag}_{i}")
""")


def test_concurrent_processes_lose_no_stage(tmp_path):
    """Real processes, real interleaving, through the file lock."""
    Manifest.open(tmp_path, {"x": 1}, "h1")
    procs = [subprocess.Popen([sys.executable, "-c", WORKER, str(SRC),
                               str(tmp_path), f"w{k}", "15"])
             for k in range(4)]
    assert all(p.wait(timeout=180) == 0 for p in procs)

    stages = _on_disk(tmp_path)["stages"]
    expected = {f"w{k}_{i}" for k in range(4) for i in range(15)}
    assert expected <= set(stages), f"lost: {sorted(expected - set(stages))}"
    assert all(stages[s]["status"] == "complete" for s in expected)
