"""The `generate(inputs_embeds=...)` contract the verbaliser depends on.

`ActivationVerbalizer._generate` assumes that generating from `inputs_embeds`
returns **only the newly generated tokens**, with no prompt ids echoed back -
there are no input ids to echo. If that assumption is wrong, every
verbalisation would be decoded with a prefix of unrelated tokens and the
explanations would be garbage in a way no later stage could detect.

Tested against a tiny randomly initialised Qwen2 on CPU: the architecture is
the one the released checkpoints use, and the contract is a property of
`transformers`, not of the weights.
"""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

VOCAB, D_MODEL, SEQ = 256, 32, 6


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=D_MODEL, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=False,
    )
    return Qwen2ForCausalLM(cfg).eval()


class TestReturnShape:
    def test_only_new_tokens_are_returned(self, tiny):
        embeds = torch.randn(1, SEQ, D_MODEL)
        attn = torch.ones(1, SEQ, dtype=torch.long)
        with torch.inference_mode():
            out = tiny.generate(inputs_embeds=embeds, attention_mask=attn,
                                max_new_tokens=5, do_sample=False, pad_token_id=0)
        # Not SEQ + 5: there are no input ids to echo back.
        assert out.shape[1] == 5, out.shape

    def test_batched(self, tiny):
        embeds = torch.randn(3, SEQ, D_MODEL)
        attn = torch.ones(3, SEQ, dtype=torch.long)
        with torch.inference_mode():
            out = tiny.generate(inputs_embeds=embeds, attention_mask=attn,
                                max_new_tokens=4, do_sample=False, pad_token_id=0)
        assert out.shape == (3, 4), out.shape

    def test_input_ids_path_does_echo(self, tiny):
        """The contrast that makes the above meaningful: with input_ids the
        prompt *is* echoed, so the two paths need different slicing."""
        ids = torch.randint(0, VOCAB, (1, SEQ))
        with torch.inference_mode():
            out = tiny.generate(input_ids=ids, max_new_tokens=5,
                                do_sample=False, pad_token_id=0)
        assert out.shape[1] == SEQ + 5


class TestInjectionRoundTrip:
    """The full verbaliser path on a tiny model: embed, inject, generate."""

    def test_injected_row_reaches_the_model(self, tiny):
        from nlaast.nla.meta import upstream

        u = upstream()
        inj, left, right = 7, 29, 102
        ids = torch.tensor([[1, left, inj, right, 2, 3]])
        emb = tiny.get_input_embeddings()(ids).float()

        vec = torch.randn(1, D_MODEL)
        scaled = u.normalize_activation(vec, 150.0)
        injected = u.inject_at_marked_positions(ids, emb, scaled, inj, left, right)

        assert torch.allclose(injected[0, 2], scaled[0], atol=1e-5)
        assert float(injected[0, 2].norm()) == pytest.approx(150.0, rel=1e-3)
        # Every other position is untouched.
        for p in (0, 1, 3, 4, 5):
            assert torch.allclose(injected[0, p], emb[0, p])

    def test_generation_from_injected_embeds_runs(self, tiny):
        from nlaast.nla.meta import upstream

        u = upstream()
        inj, left, right = 7, 29, 102
        ids = torch.tensor([[1, left, inj, right, 2, 3]])
        emb = tiny.get_input_embeddings()(ids).float()
        injected = u.inject_at_marked_positions(
            ids, emb, u.normalize_activation(torch.randn(1, D_MODEL), 150.0),
            inj, left, right,
        )
        attn = torch.ones(injected.shape[:2], dtype=torch.long)
        with torch.inference_mode():
            out = tiny.generate(inputs_embeds=injected.to(torch.float32),
                                attention_mask=attn, max_new_tokens=3,
                                do_sample=False, pad_token_id=0)
        assert out.shape == (1, 3)

    def test_a_different_vector_changes_the_output_distribution(self, tiny):
        """If injection were a no-op the verbalisation would be the same
        regardless of the activation - the failure this study could not
        otherwise detect."""
        from nlaast.nla.meta import upstream

        u = upstream()
        inj, left, right = 7, 29, 102
        ids = torch.tensor([[1, left, inj, right, 2, 3]])
        emb = tiny.get_input_embeddings()(ids).float()
        attn = torch.ones(1, ids.shape[1], dtype=torch.long)

        logits = []
        for seed in (0, 1):
            torch.manual_seed(seed)
            v = u.normalize_activation(torch.randn(1, D_MODEL), 150.0)
            injected = u.inject_at_marked_positions(ids, emb, v, inj, left, right)
            with torch.inference_mode():
                logits.append(tiny(inputs_embeds=injected.float(),
                                   attention_mask=attn).logits[0, -1])
        assert not torch.allclose(logits[0], logits[1], atol=1e-4)
