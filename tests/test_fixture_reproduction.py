"""End-to-end checks of the extended-forcing analysis against independent
references (D82, D85). Integration tests: they need the stored `main` run.

They build two *fixtures* (code tests, never data) in a temporary directory:

* identity: every "256-token" answer equals the registered 48-token answer, so
  every O256 cell must reproduce the registered analysis exactly;
* final-answer: every cut-off answer is the trace's final answer, so the O256
  cell must reproduce D79's independently computed committed-to-final
  scenario exactly.

The stored run is read in place and never written: the analysis writes only to
the temporary fixture directory. Point at another checkout's runs with
NLAAST_TEST_RUNS (default: this repository's runs/).

    pytest -m integration tests/test_fixture_reproduction.py
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("NLAAST_TEST_RUNS", ROOT / "runs"))
SCRIPTS = ROOT / "scripts"
NEEDED = [RUNS / "main" / "traces" / "traces.jsonl", RUNS / "main_forced" / "forced.jsonl",
          RUNS / "main_o3_registered" / "o3_registered.json",
          RUNS / "main_position_audit" / "position_truncation.json"]

if not all(p.exists() for p in NEEDED):
    pytest.skip("needs the stored main run (set NLAAST_TEST_RUNS)", allow_module_level=True)

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _fixture(out: Path, mode: str) -> None:
    import argparse

    from _stage import load_config
    from extended_forcing_analysis import tokenizer
    from force_extended import CHECK_SEED, N_CHECK, generated_part, menu_of, registered_completions, select

    from nlaast.logging_utils import read_jsonl

    cfg = load_config(argparse.Namespace(config="main", overrides=[], run_id=str(RUNS / "main")))
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    reg = registered_completions(traces, RUNS / "main_forced")
    cut, check = select(menu_of(cfg), reg, N_CHECK, CHECK_SEED)
    tok = tokenizer(cfg)
    suffix = cfg.generation.force_answer_suffix
    out.mkdir(parents=True)
    with open(out / "forced_ext.jsonl", "w", encoding="utf-8") as fh:
        for purpose, sel in (("cut_off", cut), ("check", check)):
            for pid in sorted(sel):
                for i in sel[pid]:
                    r = reg[(pid, i)]
                    n = len(tok(generated_part(r["text"], suffix), add_special_tokens=False)["input_ids"])
                    ans = traces[pid].get("final_answer") if (mode == "final" and purpose == "cut_off") \
                        else r["answer"]
                    fh.write(json.dumps({
                        "id": f"{pid}|{i}|{purpose}", "problem_id": pid, "index": i, "purpose": purpose,
                        "answer": ans, "text": r["text"], "finished": bool(r["finished"]) or mode == "final",
                        "n_tokens": n, "matches_final": False, "registered_answer": r["answer"],
                        "registered_finished": r["finished"], "registered_retokenised_tokens": n,
                        "begins_with_registered": True, "agrees_with_registered": True,
                        "prefix_tokens": 0, "block": []}) + "\n")
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    (out / "provenance.json").write_text(json.dumps({
        "FIXTURE": "code test, not data", "traces_sha256": sha(traces_path),
        "registered_forced_sha256": sha(RUNS / "main_forced" / "forced.jsonl")}), encoding="utf-8")


def _analyse(ext_dir: Path) -> dict:
    cmd = [sys.executable, str(SCRIPTS / "extended_forcing_analysis.py"), "--config", "main",
           "--run-id", str(RUNS / "main"), "--forced-dir", str(RUNS / "main_forced"),
           "--ext-dir", str(ext_dir)]
    env = {**os.environ, "PYTHONUTF8": "1"}
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return json.loads((ext_dir / "analysis.json").read_text(encoding="utf-8"))


def _same(a, b):
    return (a != a and b != b) or abs(a - b) <= 1e-9


@pytest.fixture(scope="module")
def identity(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx") / "identity"
    _fixture(d, "identity")
    return _analyse(d)


@pytest.fixture(scope="module")
def final(tmp_path_factory):
    d = tmp_path_factory.mktemp("fx") / "final"
    _fixture(d, "final")
    return _analyse(d)


def test_identity_fixture_reproduces_the_registered_analysis(identity):
    reg = json.loads((RUNS / "main_position_audit" / "position_truncation.json")
                     .read_text(encoding="utf-8"))["registered_outcome"]["gaps"]
    assert identity["reproduced_registered"] is True
    for k, v in identity["cells"]["O48/registered"]["gaps"].items():
        assert _same(v["gap"], reg[k]["gap"]) and v["ci95"] == reg[k]["ci95"], k
    for a, b in (("O48/registered", "O256/registered"), ("O48/net", "O256/net")):
        for k in identity["cells"][a]["gaps"]:
            assert _same(identity["cells"][a]["gaps"][k]["gap"], identity["cells"][b]["gaps"][k]["gap"]), (a, k)


def test_final_answer_fixture_reproduces_the_committed_to_final_scenario(final):
    scen = json.loads((RUNS / "main_position_audit" / "position_truncation.json")
                      .read_text(encoding="utf-8"))["truncation_scenarios"]["committed_to_final"]
    cell = final["cells"]["O256/registered"]
    assert _same(cell["oracle_bound"], scen["oracle_bound"])
    for k, v in cell["gaps"].items():
        assert _same(v["gap"], scen["gaps"][k]["gap"]) and v["ci95"] == scen["gaps"][k]["ci95"], k
    assert final["cut_off_meaning"]["f_256_equals_final"] == 1.0
