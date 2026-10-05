"""The target model: generation, token alignment and residual-stream access.

This wraps ``Qwen2.5-7B-Instruct`` - the model the released autoencoder is bound
to. Three things matter here and nothing else does:

1. **Token alignment.** Chunk boundaries are character positions in generated
   text; activations are read at token positions. The mapping between them is
   built from the generated token ids by incremental decoding, never by
   re-tokenising the decoded string. Re-tokenisation is not guaranteed to
   reproduce the sampled token sequence, and a one-token slip would read the
   residual stream at the wrong place with no visible symptom.

2. **Extraction convention.** Layer K means the *output* of decoder block K
   (``model.model.layers[K]``, first tuple element), which equals HF's
   ``hidden_states[K+1]``. This is the convention the released autoencoder was
   trained under (``nla/datagen/extractors.py`` upstream) and getting it off by
   one would feed the verbaliser vectors from the wrong layer.

3. **Survival on 6.44 GB.** Batched generation backs off on OOM rather than
   dying, because a crash eight hours into a run is expensive.
"""

from __future__ import annotations

import contextlib
import gc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from ..config import Config
from ..logging_utils import get
from . import loading

log = get(__name__)


@dataclass
class Generation:
    """One completion plus everything needed to align it to activations."""

    text: str
    token_ids: list[int]
    #: ``(char_start, char_end)`` in ``text`` for each generated token.
    offsets: list[tuple[int, int]]
    prompt_tokens: int
    finished: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)


class TargetModel:
    """Loaded target model. One instance owns one set of GPU weights."""

    def __init__(self, cfg: Config, model_path: Path | str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.tcfg = cfg.target
        self.device = self.tcfg.device if torch.cuda.is_available() else "cpu"

        if model_path is None:
            resolved = loading.ensure_nf4(self.tcfg.repo_id, self.tcfg.revision)
            model_path = resolved.local_path
            self.revision = resolved.revision
        else:
            self.revision = self.tcfg.revision
        self.model_path = Path(model_path)

        log.info("loading target model from %s (%s)", self.model_path, self.tcfg.precision)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding: with right padding, batched generation appends new
        # tokens after the pad run and the sequences desynchronise.
        self.tokenizer.padding_side = "left"

        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        if self.tcfg.precision == "nf4" and self.device != "cpu":
            # A saved NF4 checkpoint carries its own quantization_config; passing
            # another one here would conflict.
            kwargs["device_map"] = {"": 0}
        elif self.device != "cpu":
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **kwargs).eval()

        self.d_model = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        if self.d_model != self.tcfg.d_model:
            raise ValueError(
                f"config says d_model={self.tcfg.d_model} but the checkpoint has "
                f"{self.d_model}; the NLA checkpoint is dimension-bound so this "
                f"mismatch would produce silent garbage"
            )
        if not 0 <= self.tcfg.layer < self.n_layers:
            raise ValueError(f"layer {self.tcfg.layer} out of range for {self.n_layers} layers")
        log.info("target ready: %d layers, d_model=%d, device=%s | %s",
                 self.n_layers, self.d_model, self.device, loading.cuda_memory())

    # -- prompting --------------------------------------------------------

    def build_prompt(self, question: str, assistant_prefix: str = "") -> str:
        """Chat-templated prompt, optionally continuing a partial answer.

        ``assistant_prefix`` is how every prefix-continuation in this study is
        done: the generation prompt is closed and the partial trace appended, so
        the model resumes mid-answer with exactly the tokens it produced before.
        """
        messages = [
            {"role": "system", "content": self.cfg.generation.system_prompt},
            {"role": "user", "content": question},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return text + assistant_prefix

    # -- generation -------------------------------------------------------

    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        seed: int | None = None,
        batch_size: int | None = None,
        stop_strings: Sequence[str] | None = None,
    ) -> list[Generation]:
        """Batched generation with OOM backoff."""
        import torch

        bs = batch_size or self.cfg.generation.batch_size
        out: list[Generation] = []
        i = 0
        while i < len(prompts):
            take = min(bs, len(prompts) - i)
            while True:
                try:
                    out.extend(
                        self._generate_batch(
                            list(prompts[i : i + take]),
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=do_sample,
                            seed=None if seed is None else seed + i,
                            stop_strings=stop_strings,
                        )
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    loading.free_cuda()
                    if take == 1:
                        raise
                    take = max(1, take // 2)
                    bs = take
                    log.warning("CUDA OOM - retrying with batch size %d", take)
            i += take
        return out

    def _generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        seed: int | None,
        stop_strings: Sequence[str] | None,
    ) -> list[Generation]:
        import torch

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tcfg.max_position_embeddings - max_new_tokens - 8,
            add_special_tokens=False,  # the chat template already supplies them
        ).to(self.model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        if stop_strings:
            gen_kwargs.update(stop_strings=list(stop_strings), tokenizer=self.tokenizer)

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            seqs = self.model.generate(**enc, **gen_kwargs)

        prompt_len = enc["input_ids"].shape[1]
        results: list[Generation] = []
        eos_ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        for row in range(seqs.shape[0]):
            new_ids = seqs[row, prompt_len:].tolist()
            finished = any(t == self.tokenizer.eos_token_id for t in new_ids)
            # Trim the trailing EOS/pad run; those tokens have no text and
            # would shift every offset computed after them.
            while new_ids and new_ids[-1] in eos_ids:
                new_ids.pop()
            text, offsets = self.decode_with_offsets(new_ids)
            results.append(
                Generation(
                    text=text,
                    token_ids=new_ids,
                    offsets=offsets,
                    prompt_tokens=int(enc["attention_mask"][row].sum().item()),
                    finished=finished,
                )
            )
        return results

    def decode_with_offsets(self, token_ids: Sequence[int]) -> tuple[str, list[tuple[int, int]]]:
        """Decode ids and give each token its character span, exactly.

        Built by incremental decode rather than ``return_offsets_mapping`` on a
        re-tokenisation, because re-tokenising a decoded string can produce a
        different token sequence (byte-level BPE merges across what used to be a
        token boundary). The study indexes activations by these offsets, so they
        have to describe the tokens the model actually emitted.
        """
        offsets: list[tuple[int, int]] = []
        prev = ""
        for i in range(len(token_ids)):
            cur = self.tokenizer.decode(token_ids[: i + 1], skip_special_tokens=True)
            offsets.append((len(prev), len(cur)))
            prev = cur
        return prev, offsets

    def force_answer(self, question: str, prefix: str, seed: int | None = None) -> str:
        """Make the model commit to an answer from a truncated prefix.

        This is the truncation-equivalence test in the AST definition. The
        suffix is appended so the model answers rather than continuing to
        reason, and the budget is tight so it cannot start a new derivation.
        """
        gcfg = self.cfg.generation
        prompt = self.build_prompt(question, prefix + gcfg.force_answer_suffix)
        gen = self.generate(
            [prompt],
            max_new_tokens=gcfg.force_answer_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            seed=seed,
            batch_size=1,
        )[0]
        return gcfg.force_answer_suffix + gen.text

    # -- activations ------------------------------------------------------

    @contextlib.contextmanager
    def capture_layer(self, layer: int | None = None) -> Iterator[dict[str, Any]]:
        """Capture the output of decoder block ``layer`` during a forward pass.

        ``store['hidden']`` is ``[batch, seq, d_model]`` in float32 on CPU. A
        hook on the single target layer is used rather than
        ``output_hidden_states=True`` because the latter keeps all 29 layers and
        there is no VRAM for that here.
        """
        import torch

        layer = self.tcfg.layer if layer is None else layer
        store: dict[str, Any] = {"hidden": None}

        def hook(_m, _inp, output):
            h = output[0] if isinstance(output, tuple) else output
            store["hidden"] = h.detach().float().cpu()

        handle = self.model.model.layers[layer].register_forward_hook(hook)
        try:
            yield store
        finally:
            handle.remove()

    def activations_at(
        self,
        question: str,
        trace_text: str,
        positions: Sequence[int],
        layer: int | None = None,
        trace_token_ids: Sequence[int] | None = None,
    ):
        """Layer-K residual-stream vectors at generated-token ``positions``.

        ``positions`` index the generated continuation, and they were computed
        against the token sequence the model actually sampled.

        **Pass ``trace_token_ids``.** Byte-level BPE does not guarantee that
        re-tokenising decoded text reproduces the sequence that produced it -
        merges can cross what used to be a token boundary - so re-tokenising
        can shift every position by one or more. Nothing downstream would
        notice: the activations would simply be read one token off, and the
        verbaliser would describe the wrong state. When the ids are supplied
        they are used verbatim and the alignment is exact by construction;
        when they are not, the text is re-tokenised and the drift is measured
        and returned so the caller can see it.
        """
        import torch

        layer = self.tcfg.layer if layer is None else layer
        prompt = self.build_prompt(question)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        retokenised = self.tokenizer(trace_text, add_special_tokens=False)["input_ids"]
        if trace_token_ids is not None:
            trace_ids = list(trace_token_ids)
            drift = len(retokenised) - len(trace_ids)
            exact = True
        else:
            trace_ids = retokenised
            drift = 0
            exact = False

        full = prompt_ids + trace_ids
        max_len = self.tcfg.max_position_embeddings
        if len(full) > max_len:
            raise ValueError(f"sequence of {len(full)} tokens exceeds {max_len}")

        ids = torch.tensor([full], device=self.model.device)
        with self.capture_layer(layer) as store, torch.inference_mode():
            self.model(input_ids=ids, use_cache=False)
        hidden = store["hidden"][0]  # [seq, d]

        base = len(prompt_ids)
        idx, clamped = [], 0
        for p in positions:
            abs_p = base + int(p)
            resolved = min(max(abs_p, base), hidden.shape[0] - 1)
            clamped += int(resolved != abs_p)
            idx.append(resolved)
        return hidden[idx].clone(), {
            "prompt_tokens": base,
            "trace_tokens": len(trace_ids),
            "layer": layer,
            "requested": list(positions),
            "resolved": idx,
            "alignment_exact": exact,
            "retokenisation_drift": drift,
            "clamped_positions": clamped,
        }

    # -- interventions ----------------------------------------------------

    @contextlib.contextmanager
    def patch_layer(
        self,
        edit: Callable[[Any], Any],
        layer: int | None = None,
    ) -> Iterator[None]:
        """Apply ``edit`` to the output of decoder block ``layer``.

        Used by the causal stage for direction ablation. ``edit`` receives and
        returns ``[batch, seq, d_model]``; the span restriction is the caller's
        business so this stays a single, testable mechanism.
        """
        layer = self.tcfg.layer if layer is None else layer

        def hook(_m, _inp, output):
            if isinstance(output, tuple):
                return (edit(output[0]),) + tuple(output[1:])
            return edit(output)

        handle = self.model.model.layers[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def close(self) -> None:
        """Release the GPU. Stages run one model at a time; this is what makes
        that possible."""
        try:
            del self.model
        except AttributeError:
            pass
        gc.collect()
        loading.free_cuda()
