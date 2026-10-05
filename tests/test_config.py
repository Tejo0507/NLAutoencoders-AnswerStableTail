"""Configuration system.

Nothing is hardcoded: no model name, layer, seed, sample count or path
scattered through the code, and every experiment saves the exact configuration
it used. These tests pin the merge order and the hashing.
"""

import pytest

from nlaast import config as cfgmod
from nlaast.config import Config, deep_merge, from_mapping, load, parse_overrides


class TestDefaults:
    def test_target_matches_the_released_checkpoint(self):
        # The autoencoder is bound to this model at this layer and is not
        # transferable; a drifted default would silently invalidate the study.
        c = Config()
        assert c.target.repo_id == "Qwen/Qwen2.5-7B-Instruct"
        assert c.target.layer == 20
        assert c.target.d_model == 3584
        assert c.nla.av_repo.endswith("L20-av")
        assert c.nla.ar_repo.endswith("L20-ar")

    def test_is_frozen(self):
        with pytest.raises(Exception):
            Config().seed = 1

    def test_sequences_are_tuples(self):
        # Mutable defaults on a hashed config would let a stage change a
        # setting mid-run.
        c = Config()
        assert isinstance(c.stages, tuple)
        assert isinstance(c.causal.dose_coefficients, tuple)


class TestMerging:
    def test_deep_merge_is_recursive(self):
        out = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 9}})
        assert out == {"a": {"x": 1, "y": 9}, "b": 3}

    def test_deep_merge_does_not_mutate(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}

    def test_overrides_parse_yaml_scalars(self):
        out = parse_overrides(["nla.n_samples=4", "target.layer=20",
                               "generation.do_sample=false",
                               "causal.dose_coefficients=[0,1]"])
        assert out["nla"]["n_samples"] == 4
        assert out["generation"]["do_sample"] is False
        assert out["causal"]["dose_coefficients"] == [0, 1]

    def test_override_needs_an_equals(self):
        with pytest.raises(ValueError):
            parse_overrides(["bad"])

    def test_unknown_key_is_rejected(self):
        # Silently ignoring a typo'd setting is how an experiment ends up not
        # being the experiment you thought you ran.
        with pytest.raises(KeyError):
            from_mapping({"nla": {"not_a_field": 1}})

    def test_unknown_section_is_rejected(self):
        with pytest.raises(KeyError):
            from_mapping({"nonsense": {}})


class TestFiles:
    def test_base_loads(self):
        c = load()
        assert c.target.layer == 20

    @pytest.mark.parametrize("overlay", ["pilot", "main", "smoke"])
    def test_overlays_load_and_shrink_appropriately(self, overlay):
        base, c = load(), load(overlay)
        assert c.run_id == overlay
        if overlay != "main":
            assert c.data.n_gsm8k <= base.data.n_gsm8k

    def test_overlay_then_cli_override_wins(self):
        c = load("pilot", ["data.n_gsm8k=3"])
        assert c.data.n_gsm8k == 3

    def test_missing_overlay_raises(self):
        with pytest.raises(FileNotFoundError):
            load("no_such_overlay")

    def test_yaml_matches_dataclass_defaults(self):
        """configs/base.yaml restates the dataclass defaults for readability.
        If the two drift, the file people read stops describing what runs."""
        from_yaml, from_code = load().to_dict(), Config().to_dict()
        differences = {
            k: (from_yaml[k], from_code[k])
            for k in from_code
            if k not in ("run_id", "description") and from_yaml[k] != from_code[k]
        }
        assert not differences, differences


class TestHashing:
    def test_hash_is_stable(self):
        assert load("pilot").hash() == load("pilot").hash()

    def test_hash_changes_with_any_setting(self):
        assert load("pilot").hash() != load("pilot", ["nla.n_samples=99"]).hash()

    def test_round_trips_through_a_mapping(self):
        c = load("pilot")
        assert from_mapping(c.to_dict()).hash() == c.hash()

    def test_save_and_reload(self, tmp_path):
        c = load("pilot")
        p = tmp_path / "c.yaml"
        cfgmod.save(c, p)
        assert load(base=p).hash() == c.hash()


class TestDerived:
    def test_stage_dir_is_created_under_the_run(self):
        c = load("smoke")
        d = c.stage_dir("unit_test_tmp")
        assert d.exists() and d.parent == c.dir
        d.rmdir()
