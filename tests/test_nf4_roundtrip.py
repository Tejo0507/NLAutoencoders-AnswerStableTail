"""NF4 conversion and reload, on the paths the real checkpoints take.

Every model in this study is used as a 4-bit NF4 conversion of a released
bf16 checkpoint, because three Qwen-7B-shaped checkpoints do not fit in
6.44 GB of VRAM any other way. Two mechanics in that path are easy to get
wrong and expensive to discover late:

* ``loading.ensure_nf4`` has to carry the NLA sidecar and the trained value
  head across to the converted directory. A converted reconstructor without
  ``value_head.safetensors`` is not a reconstructor, and upstream asserts on it.
* upstream's ``NLACritic`` loads its backbone with no ``device_map`` and then
  calls ``.to(device)``. ``transformers`` refuses ``.to()`` on a bitsandbytes
  model when a dtype is involved, and the rules have changed across versions,
  so whether a *pre-quantised* checkpoint survives that call is a property of
  the installed stack rather than of this code.

Both are checked here at 64 dimensions, which is the same code path the 7B
checkpoints take and a few megabytes instead of 26 GB.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np

import tiny_fixtures

pytestmark = pytest.mark.slow


def _convert(src_dir, out_dir):
    """The conversion ``loading.ensure_nf4`` performs, on a local directory.

    ``ensure_nf4`` resolves its source through the HuggingFace hub, which a
    local fixture has no entry in, so the same three calls are made directly:
    load under the project's own NF4 config, save, and carry the sidecar files
    across. Keeping the quantisation config shared is the part that matters -
    a test that quantised differently from the pipeline would prove nothing.
    """
    import shutil

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nlaast.models import loading

    model = AutoModelForCausalLM.from_pretrained(
        str(src_dir),
        quantization_config=loading.nf4_config(),
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(str(src_dir)).save_pretrained(out_dir)
    for extra in ("nla_meta.yaml", "value_head.safetensors", "chat_template.jinja"):
        p = src_dir / extra
        if p.exists():
            shutil.copy2(p, out_dir / extra)
    del model
    loading.free_cuda()
    return out_dir


@pytest.fixture(scope="module")
def bf16_fixtures(tmp_path_factory):
    if not torch.cuda.is_available():
        pytest.skip("NF4 conversion needs CUDA")
    pytest.importorskip("bitsandbytes")
    tok_src = tiny_fixtures.tokenizer_source()
    if tok_src is None:
        pytest.skip("no local Qwen2.5 tokeniser")
    src = tmp_path_factory.mktemp("bf16")
    tiny_fixtures.build_checkpoints(src, tok_src)
    return src


@pytest.fixture(scope="module")
def nf4_ar(bf16_fixtures, tmp_path_factory):
    return _convert(bf16_fixtures / "nlaast-fixture__ar",
                    tmp_path_factory.mktemp("nf4_ar") / "ar")


@pytest.fixture(scope="module")
def nf4_av(bf16_fixtures, tmp_path_factory):
    return _convert(bf16_fixtures / "nlaast-fixture__av",
                    tmp_path_factory.mktemp("nf4_av") / "av")


class TestConvertedCheckpoint:
    def test_the_sidecar_and_value_head_travel_with_it(self, nf4_ar):
        """Without these two files the converted directory is unusable, and
        nothing downstream of the conversion would say why."""
        assert (nf4_ar / "nla_meta.yaml").exists()
        assert (nf4_ar / "value_head.safetensors").exists()
        assert (nf4_ar / "config.json").exists()

    def test_it_is_really_quantised(self, nf4_ar):
        import json

        cfg = json.loads((nf4_ar / "config.json").read_text(encoding="utf-8"))
        qc = cfg.get("quantization_config", {})
        assert qc.get("quant_method") == "bitsandbytes_4bit" or qc.get("load_in_4bit")
        assert qc.get("bnb_4bit_quant_type") == "nf4"


class TestVerbaliserOnNf4:
    """The riskiest mechanism in the study, on a 4-bit checkpoint.

    Upstream drives the verbaliser through an SGLang server; this project
    substitutes a local ``transformers`` path that injects into
    ``inputs_embeds``. Whether that survives a bitsandbytes model - whose
    ``nn.Embedding`` is *not* quantised while its linears are - is a property
    of the stack, and the real 7B AV costs 15 GB to find out.
    """

    def test_injection_and_generation_work_through_a_quantised_model(self, nf4_av):
        from nlaast.config import load as load_config
        from nlaast.nla.verbalizer import ActivationVerbalizer

        cfg = load_config("tiny", ["nla.device=cuda", "nla.max_new_tokens=8"])
        av = ActivationVerbalizer(cfg, nf4_av)
        try:
            assert av.tokenizer_report["ok"], av.tokenizer_report
            # The embedding table must still be floating point: the injected
            # vector enters through it, so a quantised table would corrupt it.
            assert av.embed.weight.dtype in (torch.bfloat16, torch.float16,
                                             torch.float32)
            assert av.embed_scale == pytest.approx(1.0)

            rng = np.random.default_rng(0)
            vectors = rng.standard_normal((2, tiny_fixtures.D_MODEL)).astype(np.float32)
            out = av.verbalise(vectors, seed=1, n_samples=1, batch_size=2)
            assert len(out) == 2
            for samples in out:
                assert len(samples) == 1
                assert samples[0].n_tokens > 0

            # The injected row must arrive at the sidecar's injection scale,
            # which is what keeps the vector in distribution.
            embeds, prompt_len = av._build_embeds(vectors)
            site = av.tokenizer_report["position"]
            for row in range(2):
                norm = float(embeds[row, site].norm())
                assert norm == pytest.approx(av.meta.injection_scale, rel=1e-2)
            assert prompt_len == embeds.shape[1]
        finally:
            av.close()

    def test_the_gaussian_control_matches_the_activation_norm(self, nf4_av):
        from nlaast.config import load as load_config
        from nlaast.nla.verbalizer import ActivationVerbalizer

        cfg = load_config("tiny", ["nla.device=cuda", "nla.max_new_tokens=4"])
        av = ActivationVerbalizer(cfg, nf4_av)
        try:
            rng = np.random.default_rng(1)
            ref = rng.standard_normal((3, tiny_fixtures.D_MODEL)).astype(np.float32) * 7.0
            noise = av.gaussian_control(ref, rng)
            assert np.allclose(np.linalg.norm(noise, axis=1),
                               np.linalg.norm(ref, axis=1), rtol=1e-4)
            out = av.verbalise(noise, seed=2, n_samples=1,
                               source="gaussian_control", batch_size=3)
            assert all(s[0].source == "gaussian_control" for s in out)
        finally:
            av.close()


class TestCriticOnNf4:
    """The call that decides whether the real AR arm can run at all."""

    def test_upstream_critic_loads_the_nf4_checkpoint_and_reconstructs(self, nf4_ar):
        from nlaast.config import load as load_config
        from nlaast.nla.reconstructor import ActivationReconstructor

        cfg = load_config("tiny", ["nla.device=cuda"])
        ar = ActivationReconstructor(cfg, nf4_ar)
        try:
            vec = ar.reconstruct("A short explanation about a committed answer.")
            assert vec.shape == (tiny_fixtures.D_MODEL,)
            assert np.isfinite(vec).all()

            gold = np.random.default_rng(0).standard_normal(
                tiny_fixtures.D_MODEL).astype(np.float32)
            r = ar.score("A short explanation about a committed answer.", gold)
            # The released convention: both vectors normalised to mse_scale,
            # so MSE and cosine are the same number twice over.
            assert abs(r.mse - 2 * (1 - r.cosine)) < 1e-3
            assert ar.mse_scale == pytest.approx(float(np.sqrt(tiny_fixtures.D_MODEL)))
        finally:
            ar.close()
