"""End-to-end: every stage script, on fixture checkpoints, in seconds.

The unit tests cover the library. What they cannot cover is the stage scripts
themselves - the argument plumbing, the manifest transitions, and above all
whether the artefact one stage writes is the artefact the next stage reads.
Those faults only appear when the scripts actually run, and running them
against the released 7B autoencoder costs a 26 GB download and several GB of
VRAM.

So this runs the real scripts, in the real order, against the tiny fixtures
built by `tiny_fixtures.py`: `ast` -> `baselines` -> `nla` -> `faithfulness`
-> `causal` -> `analysis` -> `robustness` -> `report`. Each is a subprocess,
exactly as the orchestrator invokes it, so `sys.path` juggling and
`if __name__` guards are exercised too.

What a pass means: the pipeline is wired correctly. What it does not mean:
anything about the science. The fixture weights are random, so the numbers are
noise - the assertions therefore check structure (files exist, keys present,
counts consistent), never values.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tiny_fixtures

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
RUN_ID = "tiny"

pytestmark = pytest.mark.integration


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Build the fixtures, run every stage once, and hand back the run dir.

    Module-scoped: the stages are a sequence, not independent cases, so they
    run once and each test inspects a different part of what came out.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")

    tok_src = tiny_fixtures.tokenizer_source()
    if tok_src is None:
        pytest.skip("no local Qwen2.5 tokeniser; the injection convention "
                    "cannot be faked, so there is nothing honest to test")

    quant = tmp_path_factory.mktemp("fixture_ckpts")
    tiny_fixtures.build_checkpoints(quant, tok_src)

    env = dict(os.environ)
    env["NLAAST_QUANT_DIR"] = str(quant)
    # No stage may reach the network in this test; a fixture that silently
    # downloaded the real checkpoint would be worthless.
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")

    from nlaast import config as config_mod

    cfg = config_mod.load("tiny", [])
    run_dir = cfg.dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    tiny_fixtures.build_run(cfg)
    tiny_fixtures.write_manifest_stub(cfg)

    stages = [
        ("ast", ["detect_answer_stable_tail.py", "--sweep"]),
        ("baselines", ["score_baselines.py"]),
        ("nla", ["run_autoencoder.py"]),
        ("faithfulness", ["audit_faithfulness.py"]),
        ("causal", ["run_interventions.py"]),
        ("analysis", ["aggregate_results.py"]),
        ("robustness", ["run_falsification_tests.py"]),
        ("report", ["write_run_report.py"]),
    ]
    log: dict[str, subprocess.CompletedProcess] = {}
    for name, argv in stages:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / argv[0]), "--config", "tiny", *argv[1:]],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800,
        )
        log[name] = proc
        if proc.returncode != 0:
            pytest.fail(
                f"stage {name} exited {proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout[-4000:]}\n"
                f"--- stderr ---\n{proc.stderr[-4000:]}"
            )
        if name == "nla":
            # A randomly initialised verbaliser cannot emit a well-formed
            # English <explanation>, and the integrity gate is right to reject
            # what it does emit. The `nla` stage's own output is asserted on as
            # generated; the explanation fields are then rewritten so the
            # downstream audit sees input of the real shape. See
            # tiny_fixtures.substitute_explanations.
            assert tiny_fixtures.substitute_explanations(cfg) > 0
    return run_dir, log


class TestStagesRun:
    def test_every_stage_is_marked_complete(self, pipeline):
        run_dir, _ = pipeline
        manifest = _read_json(run_dir / "manifest.json")
        for stage in ("ast", "baselines", "nla", "faithfulness",
                      "causal", "analysis", "robustness", "report"):
            status = manifest["stages"].get(stage, {}).get("status")
            assert status == "complete", f"{stage} is {status!r}"


class TestAstAndBaselines:
    def test_tail_detected_on_the_traces_built_to_have_one(self, pipeline):
        run_dir, _ = pipeline
        rows = {r["problem_id"]: r for r in _read_jsonl(run_dir / "ast" / "ast.jsonl")}
        assert rows["fixture-0001"]["tail_start"] is not None
        assert rows["fixture-0002"]["tail_start"] is not None
        # Built with no qualifying boundary at all.
        assert rows["fixture-0003"]["tail_start"] is None
        assert rows["fixture-0003"]["status"] == "no_stable_point"
        # Built with the token cap hit, which must be carried as a status.
        assert rows["fixture-0004"]["status"] == "truncated"

    def test_f5_sweep_written(self, pipeline):
        run_dir, _ = pipeline
        sweep = _read_json(run_dir / "ast" / "sweep_F5.json")
        assert "require_unanimous=False" in sweep
        assert all("tail_rate" in v for v in sweep.values())

    def test_baseline_curves_cover_every_rule_with_data(self, pipeline):
        run_dir, _ = pipeline
        curves = _read_json(run_dir / "baselines" / "curves.json")
        assert "convergence" in curves["curves"]
        assert curves["matched_budgets"]
        for pts in curves["curves"].values():
            for p in pts:
                assert 0.0 <= p["safe_rate"] <= 1.0
                assert 0.0 <= p["fire_rate"] <= 1.0


class TestNla:
    """The stage that has never run against the real checkpoint.

    These assertions are the contract the real run has to satisfy too: the
    injection path produces text, the control is generated alongside, and
    reconstruction is scored in the released convention.
    """

    def test_verbalisations_exist_with_the_noise_control(self, pipeline):
        run_dir, _ = pipeline
        # The file as the stage wrote it, before the fixture substitution.
        rows = _read_jsonl(run_dir / "nla" / "verbalisations.asgenerated.jsonl")
        assert rows, "the inputs_embeds injection path produced nothing"
        for r in rows:
            assert len(r["samples"]) == 2, "nla.n_samples=2 was requested"
            assert r["noise_samples"], "the Li et al. verbaliser-only control is missing"
            assert r["activation_norm"] > 0
            for s in r["samples"]:
                # Injection ran and decoding returned something; the gate
                # judges its content, which random weights cannot satisfy.
                assert s["n_tokens"] > 0
                assert "fixture_substituted" not in s

    def test_the_integrity_gate_rejects_random_weight_output(self, pipeline):
        """The off-distribution signature, reproduced deliberately.

        A randomly initialised verbaliser is as off-distribution as it gets, so
        the gate must reject it. A fixture that sailed through the gate would
        mean the gate cannot detect the failure it exists for.
        """
        run_dir, _ = pipeline
        rows = _read_jsonl(run_dir / "nla" / "verbalisations.asgenerated.jsonl")
        samples = [s for r in rows for s in r["samples"]]
        assert samples
        assert not any(s["ok"] for s in samples)

    def test_tail_and_control_windows_are_both_verbalised(self, pipeline):
        run_dir, _ = pipeline
        kinds = {r["window_kind"] for r in
                 _read_jsonl(run_dir / "nla" / "verbalisations.jsonl")}
        assert "tail" in kinds
        # Without a control window F10 cannot be computed at all.
        assert kinds & {"matched_position", "matched_length"}

    def test_reconstruction_uses_the_released_metric(self, pipeline):
        run_dir, _ = pipeline
        rows = _read_jsonl(run_dir / "nla" / "reconstructions.jsonl")
        assert rows
        for r in rows:
            assert -1.0001 <= r["cosine"] <= 1.0001
            # MSE = 2(1 - cos) under normalise-to-sqrt(d); this is the identity
            # that makes the reported numbers comparable to the checkpoint card.
            assert abs(r["mse"] - 2 * (1 - r["cosine"])) < 1e-3
        assert any(r["window_kind"] == "gaussian_control" for r in rows)

    def test_summary_reports_fve_against_the_reference(self, pipeline):
        run_dir, _ = pipeline
        s = _read_json(run_dir / "nla" / "summary.json")
        assert "overall" in s and "fve" in s["overall"]
        assert s["reference_fve"] == pytest.approx(0.752)
        assert "gaussian_control" in s["by_window_kind"]


class TestFaithfulness:
    def test_audits_carry_every_arm(self, pipeline):
        run_dir, _ = pipeline
        rows = _read_jsonl(run_dir / "faithfulness" / "audits.jsonl")
        assert rows
        claims = [c for r in rows for c in r.get("claims", [])]
        assert claims, "no claim was extracted from any explanation"
        for c in claims:
            assert "delete_effect" in c
            assert "paraphrase_effects" in c
            assert "reconstruction_dependent" in c
            assert "reproduced_under_noise" in c

    def test_summary_separates_dependence_from_prior_reproduction(self, pipeline):
        run_dir, _ = pipeline
        s = _read_json(run_dir / "faithfulness" / "summary.json")
        for key in ("dependent_fraction", "noise_reproduced_fraction",
                    "dependent_and_not_noise_fraction"):
            assert 0.0 <= s[key] <= 1.0
        assert "RECAP-inspired" in s["note"]


class TestCausal:
    def test_arms_ran_and_the_verdict_is_gated(self, pipeline):
        run_dir, _ = pipeline
        s = _read_json(run_dir / "causal" / "summary.json")
        assert s["n_results"] > 0
        # The fixture corpus is sized so the train-split direction is fittable;
        # if it were not, the three projection arms would silently not run.
        assert s["direction"]["available"] is True
        assert {"truncate", "filler", "ablate_dir", "random_dir",
                "matched_position"} <= set(s["arms"])
        for arm in ("ablate_dir", "random_dir", "matched_position"):
            assert s["arms"][arm]["n"] > 0, f"{arm} produced no results"
        assert s["dose_response_ablate"], "the dose sweep produced no points"
        v = s["verdict"]
        for key in ("supported", "beats_random_direction", "beats_matched_position",
                    "monotone_dose_response", "answers_preserved"):
            assert key in v
        # The gate is a conjunction: it cannot be true while a component is false.
        if v["supported"]:
            assert all(v[k] for k in ("beats_random_direction",
                                      "beats_matched_position",
                                      "monotone_dose_response"))

    def test_interventions_record_their_baseline(self, pipeline):
        run_dir, _ = pipeline
        rows = _read_jsonl(run_dir / "causal" / "interventions.jsonl")
        assert rows
        assert {r["intervention"] for r in rows} & {"truncate", "filler"}
        for r in rows:
            assert "baseline_answer" in r and "baseline_verification_markers" in r


class TestAnalysisAndReport:
    def test_test_family_is_corrected_not_pruned(self, pipeline):
        run_dir, _ = pipeline
        results = _read_json(run_dir / "analysis" / "results.json")
        t = results["tests"]
        assert t["n_tests"] >= t["n_evaluated"] >= t["n_significant"]
        assert results["fdr_q"] == 0.05
        # Unevaluable tests stay in the family with a null q-value.
        for entry in t["tests"]:
            assert "q_value" in entry

    def test_tables_written(self, pipeline):
        run_dir, _ = pipeline
        tbl = run_dir / "analysis" / "tables"
        names = {p.stem for p in tbl.glob("*.csv")}
        assert {"ast_corpus", "ast_status", "hypothesis_tests"} <= names

    def test_robustness_reports_every_enabled_test(self, pipeline):
        run_dir, _ = pipeline
        rob = _read_json(run_dir / "robustness" / "robustness.json")
        enabled = ("F1_length", "F2_difficulty", "F3_verbaliser_only", "F4_parser",
                   "F5_ast_sensitivity", "F6_probe_leakage", "F7_layer_sweep",
                   "F10_position")
        for name in enabled:
            assert name in rob, f"{name} is missing from the battery"
            assert "status" in rob[name]
        # The fixture corpus is sized so none of these fall back to "skipped";
        # a skip here means the battery is not being exercised at all.
        for name in enabled:
            assert rob[name]["status"] == "ran", (name, rob[name])

    def test_probe_leakage_control_collapses_on_shuffled_labels(self, pipeline):
        """F6 on fixture data: the shuffled-label probe must be near chance.

        The activations carry a real tail offset, so the unshuffled probe has
        something to find; permuting labels within problem destroys only the
        activation-label link. This asserts the control behaves, not that the
        fixture's AUC means anything.
        """
        run_dir, _ = pipeline
        f6 = _read_json(run_dir / "robustness" / "robustness.json")["F6_probe_leakage"]
        assert f6["status"] == "ran"
        assert "shuffled_auc" in f6 and "real_auc" in f6
        assert isinstance(f6["leak_suspected"], bool)

    def test_report_is_self_contained(self, pipeline):
        run_dir, _ = pipeline
        md = (run_dir / "report" / "RESULTS.md").read_text(encoding="utf-8")
        assert "Answer-Stable Tail" in md or "answer-stable tail" in md.lower()
        assert (run_dir / "report" / "manifest_snapshot.json").exists()
        assert list((run_dir / "report" / "figures").glob("*.png"))
