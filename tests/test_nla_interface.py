"""The NLA interface: vendored upstream, sidecar parsing, integrity gating.

Nothing here loads a checkpoint. What it pins is the set of conventions that
would otherwise fail silently — upstream's own docstrings warn that a wrong
injection position produces CJK output rather than an error, and that a
mismatched checkpoint produces confident nonsense rather than an exception.
"""

import math

import numpy as np
import pytest
import torch
import yaml

from nlaast.nla.meta import (
    chat_template_ids,
    check_consistency,
    load_meta,
    upstream,
    verify_tokenizer,
)
from nlaast.nla.verbalizer import score_integrity

#: The released kitft/nla-qwen2.5-7b-L20-av sidecar, with one abbreviation:
#: every constant below - injection character, token id, both neighbour ids,
#: injection scale, MSE scale, extraction layer, and the AR template - is
#: verbatim and verified against the real file. The **AV template** is
#: shortened to its `<concept>{injection_char}</concept>` core; the released
#: one wraps the same marker in a 125-token researcher-persona prompt. The
#: marker and its immediate neighbours are what the injection convention
#: depends on, and they are identical in both.
AV_SIDECAR = {
    "kind": "nla_model", "schema_version": 2, "role": "av", "d_model": 3584,
    "extraction": {"injection_scale": 150.0, "mse_scale": 59.86651818838306},
    "tokens": {"injection_char": "㈎", "injection_token_id": 149705,
               "injection_left_neighbor_id": 29, "injection_right_neighbor_id": 522,
               "critic_suffix_ids": None},
    "prompt_templates": {
        "av": "Here is the vector:\n\n<concept>{injection_char}</concept>\n",
        "ar": "Summary of the following text: <text>{explanation}</text> <summary>",
    },
    "extraction_layer_index": 20,
}

AR_SIDECAR = {
    **AV_SIDECAR, "role": "ar",
    "extraction": {"injection_scale": None, "mse_scale": 59.86651818838306},
    "tokens": {**AV_SIDECAR["tokens"], "critic_suffix_ids": [1318, 29, 366, 1708, 29]},
}


def write_sidecar(tmp_path, data, name="ckpt"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "nla_meta.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return d


class _FakeTokenizer:
    """Returns chat-template ids in one of the shapes transformers has used.

    ``apply_chat_template(tokenize=True)`` returned a bare ``list[int]`` in
    transformers 4.x and returns a ``BatchEncoding`` in 5.x. The injection-site
    check iterates the result, so on 5.x it iterated dictionary *keys*, found
    no injection site, and refused to run the verbaliser with a "tokenizer
    drift" error that was not true. These pin the normalisation.
    """

    def __init__(self, ids, shape):
        self._ids, self._shape = ids, shape

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        assert tokenize and add_generation_prompt
        if self._shape == "list":
            return list(self._ids)
        if self._shape == "batch_encoding":
            return {"input_ids": list(self._ids), "attention_mask": [1] * len(self._ids)}
        if self._shape == "nested":
            return {"input_ids": [list(self._ids)], "attention_mask": [[1] * len(self._ids)]}
        if self._shape == "tensor":
            return {"input_ids": torch.tensor([self._ids]), "attention_mask": None}
        raise AssertionError(self._shape)


IDS = [11, 12, 29, 149705, 522, 13]


class TestChatTemplateIds:
    @pytest.mark.parametrize("shape", ["list", "batch_encoding", "nested", "tensor"])
    def test_every_return_shape_flattens_to_the_same_ids(self, shape):
        assert chat_template_ids(_FakeTokenizer(IDS, shape), "x") == IDS

    @pytest.mark.parametrize("shape", ["list", "batch_encoding", "nested", "tensor"])
    def test_verify_tokenizer_finds_the_site_in_every_shape(self, shape, tmp_path):
        meta = load_meta(write_sidecar(tmp_path, AV_SIDECAR, f"av_{shape}"))
        report = verify_tokenizer(_FakeTokenizer(IDS, shape), meta)
        assert report["ok"], report
        assert report["position"] == 3
        assert (report["left"], report["right"]) == (29, 522)

    def test_a_genuinely_drifted_neighbour_is_still_caught(self, tmp_path):
        meta = load_meta(write_sidecar(tmp_path, AV_SIDECAR, "av_drift"))
        bad = [11, 12, 999, 149705, 522, 13]
        report = verify_tokenizer(_FakeTokenizer(bad, "batch_encoding"), meta)
        assert not report["ok"]
        assert "neighbour drift" in report["error"]


def _released_sidecar(role: str):
    """The real checkpoint's directory, if its small files are cached locally.

    `nla_meta.yaml`, the tokeniser and the configs are a few tens of megabytes
    and can be fetched without the 26 GB of weights:

        snapshot(repo, allow_patterns=["*.json", "*.yaml", "*.jinja", "*.txt"])

    When they are present the compatibility gate can be checked against the
    conventions the released checkpoints actually ship, rather than against a
    sidecar this repository wrote. Returns ``None`` otherwise so the test
    skips rather than fails on a fresh clone.
    """
    from nlaast import paths

    hub = paths.model_cache() / "hub"
    if not hub.is_dir():
        return None
    for repo in sorted(hub.glob(f"models--kitft--nla-qwen2.5-7b-L20-{role}")):
        for snap in sorted(repo.glob("snapshots/*")):
            if (snap / "nla_meta.yaml").exists():
                return snap
    return None


@pytest.fixture(scope="module")
def released():
    av, ar = _released_sidecar("av"), _released_sidecar("ar")
    if av is None or ar is None:
        pytest.skip("released sidecars are not cached locally")
    return load_meta(av), load_meta(ar), av


class TestAgainstTheReleasedSidecars:
    """The compatibility gate, run against the real conventions.

    The autoencoders are bound to one model at one layer and a mismatch
    produces confident nonsense rather than an error, so this gate is the last
    line of defence before 26 GB of weights are loaded and believed. Checking
    it against the shipped sidecars - not against a copy in this repository -
    is the only version of the check that could catch a drift in either.
    """

    def test_the_pair_is_mutually_consistent_with_the_configured_target(self, released):
        from nlaast import config as config_mod

        av, ar, _ = released
        cfg = config_mod.load("pilot", [])
        report = check_consistency(av, ar, cfg.target.layer, cfg.target.d_model)
        assert report["ok"], report["problems"]

    def test_the_constants_this_project_pinned_are_the_shipped_ones(self, released):
        av, ar, _ = released
        assert av.d_model == 3584
        assert av.extraction_layer_index == 20
        assert av.injection_token_id == 149705
        assert (av.injection_left_neighbor_id, av.injection_right_neighbor_id) == (29, 522)
        assert av.injection_scale == pytest.approx(150.0)
        assert ar.mse_scale == pytest.approx(math.sqrt(3584))
        assert ar.ar_template == AR_SIDECAR["prompt_templates"]["ar"]

    def test_the_live_tokeniser_reproduces_the_injection_site(self, released):
        """With the released AV prompt, which is far longer than the fixture's
        and places the marker deep inside it."""
        from transformers import AutoTokenizer

        av, _, av_dir = released
        tok = AutoTokenizer.from_pretrained(str(av_dir))
        report = verify_tokenizer(tok, av)
        assert report["ok"], report
        assert report["n_injection_sites"] == 1
        assert (report["left"], report["right"]) == (29, 522)
        assert report["position"] > 0


class TestVendoredUpstream:
    """The injection arithmetic is upstream's, not reimplemented. These tests
    confirm the vendored copy is present and behaves as the study assumes."""

    def test_exports_the_functions_this_project_depends_on(self):
        u = upstream()
        for name in ("normalize_activation", "inject_at_marked_positions",
                     "NLACritic", "resolve_embed_scale", "load_nla_config"):
            assert hasattr(u, name), name

    def test_normalize_activation_sets_the_l2_norm(self):
        u = upstream()
        v = torch.randn(1, 3584) * 37.0
        for scale in (150.0, math.sqrt(3584)):
            assert float(u.normalize_activation(v, scale).norm()) == pytest.approx(
                scale, rel=1e-4)

    def test_normalize_preserves_direction(self):
        u = upstream()
        v = torch.randn(1, 64)
        out = u.normalize_activation(v, 150.0)
        cos = float((v @ out.T) / (v.norm() * out.norm()))
        assert cos == pytest.approx(1.0, abs=1e-5)

    def test_injection_requires_the_neighbours_to_match(self):
        """The neighbour check is what rejects a stray marker character in
        ordinary text. Without it the vector lands in the wrong position."""
        u = upstream()
        ids = torch.tensor([[1, 29, 149705, 522, 2]])
        emb = torch.zeros(1, 5, 8)
        vec = torch.ones(1, 8) * 3.0
        out = u.inject_at_marked_positions(ids, emb, vec, 149705, 29, 522)
        assert torch.allclose(out[0, 2], torch.full((8,), 3.0))
        assert torch.allclose(out[0, 1], torch.zeros(8))

    def test_injection_rejects_a_wrong_neighbour(self):
        u = upstream()
        ids = torch.tensor([[1, 99, 149705, 522, 2]])   # left neighbour wrong
        with pytest.raises((RuntimeError, AssertionError)):
            u.inject_at_marked_positions(ids, torch.zeros(1, 5, 8),
                                         torch.ones(1, 8), 149705, 29, 522)

    def test_injection_does_not_mutate_its_input(self):
        u = upstream()
        emb = torch.zeros(1, 5, 8)
        u.inject_at_marked_positions(torch.tensor([[1, 29, 149705, 522, 2]]),
                                     emb, torch.ones(1, 8), 149705, 29, 522)
        assert torch.allclose(emb, torch.zeros(1, 5, 8))


class TestSidecar:
    def test_parses_the_av(self, tmp_path):
        m = load_meta(write_sidecar(tmp_path, AV_SIDECAR, "av"))
        assert m.role == "av"
        assert m.d_model == 3584
        assert m.extraction_layer_index == 20
        assert m.injection_token_id == 149705
        assert m.injection_left_neighbor_id == 29
        assert m.injection_right_neighbor_id == 522
        assert m.injection_scale == 150.0

    def test_parses_the_ar(self, tmp_path):
        m = load_meta(write_sidecar(tmp_path, AR_SIDECAR, "ar"))
        assert m.role == "ar"
        assert m.injection_scale is None
        assert m.critic_suffix_ids == (1318, 29, 366, 1708, 29)
        assert "{explanation}" in m.ar_template

    def test_mse_scale_is_sqrt_d_model(self, tmp_path):
        # Upstream's choice: with both vectors at L2 = sqrt(d), MSE = 2(1-cos).
        m = load_meta(write_sidecar(tmp_path, AR_SIDECAR, "ar"))
        assert m.mse_scale == pytest.approx(math.sqrt(m.d_model), abs=1e-6)

    def test_missing_sidecar_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_meta(tmp_path / "nothing")


class TestConsistency:
    def _meta(self, tmp_path, data, name):
        return load_meta(write_sidecar(tmp_path, data, name))

    def test_matching_pair_passes(self, tmp_path):
        av = self._meta(tmp_path, AV_SIDECAR, "av")
        ar = self._meta(tmp_path, AR_SIDECAR, "ar")
        assert check_consistency(av, ar, 20, 3584)["ok"]

    def test_wrong_layer_is_caught(self, tmp_path):
        # The autoencoder is bound to one model at one layer and does not
        # transfer; this mismatch would otherwise produce plausible nonsense.
        av = self._meta(tmp_path, AV_SIDECAR, "av")
        ar = self._meta(tmp_path, AR_SIDECAR, "ar")
        r = check_consistency(av, ar, 24, 3584)
        assert not r["ok"] and any("layer" in p for p in r["problems"])

    def test_wrong_d_model_is_caught(self, tmp_path):
        av = self._meta(tmp_path, AV_SIDECAR, "av")
        ar = self._meta(tmp_path, AR_SIDECAR, "ar")
        r = check_consistency(av, ar, 20, 4096)
        assert not r["ok"] and any("d_model" in p for p in r["problems"])

    def test_mismatched_pair_is_caught(self, tmp_path):
        av = self._meta(tmp_path, AV_SIDECAR, "av")
        other = {**AR_SIDECAR, "extraction_layer_index": 32,
                 "tokens": {**AR_SIDECAR["tokens"], "injection_token_id": 999}}
        ar = self._meta(tmp_path, other, "ar2")
        r = check_consistency(av, ar, 20, 3584)
        assert not r["ok"]
        assert len(r["problems"]) >= 2


class TestIntegrityGate:
    """Upstream documents the off-distribution failure signature: the
    verbaliser emits CJK instead of an English <explanation>."""

    def test_accepts_a_well_formed_english_explanation(self):
        r = score_integrity(
            "<explanation>The vector relates to arithmetic word problems "
            "involving money.</explanation>", 0.75)
        assert r["ok"] and r["well_formed"] and r["cjk_chars"] == 0
        assert r["explanation"].startswith("The vector")

    def test_rejects_cjk_output(self):
        r = score_integrity("<explanation>这是一个关于数学的向量描述内容</explanation>", 0.75)
        assert not r["ok"]
        assert r["cjk_chars"] > 0

    def test_rejects_missing_tags(self):
        r = score_integrity("The vector relates to arithmetic.", 0.75)
        assert not r["ok"] and not r["well_formed"]

    def test_rejects_empty_explanation(self):
        assert not score_integrity("<explanation></explanation>", 0.75)["ok"]

    def test_threshold_is_configurable(self):
        mixed = "<explanation>mostly english text here 这是</explanation>"
        assert not score_integrity(mixed, 0.95)["ok"]

    def test_only_the_explanation_body_is_scored(self):
        # Preamble outside the tags must not drag the ASCII fraction around.
        r = score_integrity(
            "Sure! <explanation>Arithmetic word problems.</explanation> Done.", 0.75)
        assert r["ok"] and r["explanation"] == "Arithmetic word problems."


class TestGaussianControl:
    def test_matches_the_reference_norm(self):
        """Li et al.'s control is only meaningful at matched norm: an off-norm
        vector would fail injection for a trivial reason and the control would
        be vacuous."""
        from nlaast.nla.verbalizer import ActivationVerbalizer

        ref = np.random.default_rng(0).standard_normal((3, 64)).astype(np.float32) * 17
        rng = np.random.default_rng(1)
        out = ActivationVerbalizer.gaussian_control(None, ref, rng)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1),
                                   np.linalg.norm(ref, axis=1), rtol=1e-5)

    def test_is_not_the_reference(self):
        from nlaast.nla.verbalizer import ActivationVerbalizer

        ref = np.ones((1, 64), dtype=np.float32)
        out = ActivationVerbalizer.gaussian_control(None, ref, np.random.default_rng(0))
        cos = float((out @ ref.T).ravel()[0]
                    / (np.linalg.norm(out) * np.linalg.norm(ref)))
        assert abs(cos) < 0.5
