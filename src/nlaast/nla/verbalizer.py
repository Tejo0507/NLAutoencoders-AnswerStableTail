"""Activation Verbaliser - vector to text.

Upstream drives the verbaliser through an SGLang server using its
``input_embeds`` API. SGLang has no Windows support and would not fit beside the
model in 6.44 GB, so this module runs the same computation through
``transformers`` instead. The arithmetic is identical and the two
correctness-critical steps - rescaling and injection - call the *vendored*
upstream functions rather than reimplementing them:

    chat template (from the sidecar)
      -> tokenise
      -> embedding lookup (x embed_scale; 1.0 for Qwen)
      -> normalize_activation(v, injection_scale)        [upstream]
      -> inject_at_marked_positions(..., left, right)    [upstream]
      -> generate(inputs_embeds=...)

The one substantive difference from upstream is the generation backend, recorded
in ``third_party/nla_inference/NOTICE.md``.

Integrity gating. Upstream documents a specific failure mode: if the injected
vector is off-distribution the verbaliser emits CJK text instead of an English
``<explanation>``. Because this project feeds it activations from a 4-bit target
model, that gate is checked on every sample and the rate is reported, not
assumed to be zero.
"""

from __future__ import annotations

import gc
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..logging_utils import get
from ..models import loading
from .meta import (
    NLAMeta,
    chat_template_ids,
    load_meta,
    upstream,
    verify_tokenizer,
)

log = get(__name__)

_EXPLANATION = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL)
_CJK = re.compile(r"[　-鿿豈-﫿＀-￯]")


@dataclass
class Verbalisation:
    """One AV sample plus the integrity evidence for it."""

    text: str
    explanation: str | None
    raw: str
    sample_index: int
    seed: int
    n_tokens: int
    #: Integrity gate (see module docstring).
    ascii_fraction: float = 1.0
    cjk_chars: int = 0
    well_formed: bool = False
    ok: bool = False
    #: ``activation`` | ``gaussian_control`` - the Li et al. verbaliser-only arm.
    source: str = "activation"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_integrity(text: str, min_ascii_fraction: float) -> dict[str, Any]:
    """Did the verbaliser produce a usable English explanation?"""
    explanation = None
    m = _EXPLANATION.search(text)
    if m:
        explanation = m.group(1).strip()
    body = explanation if explanation else text
    cjk = len(_CJK.findall(body))
    printable = [c for c in body if not c.isspace()]
    ascii_fraction = (
        sum(1 for c in printable if ord(c) < 128) / len(printable) if printable else 0.0
    )
    well_formed = explanation is not None and len(explanation) > 0
    return {
        "explanation": explanation,
        "cjk_chars": cjk,
        "ascii_fraction": ascii_fraction,
        "well_formed": well_formed,
        "ok": bool(well_formed and ascii_fraction >= min_ascii_fraction and cjk == 0),
    }


class ActivationVerbalizer:
    """Loaded AV checkpoint."""

    def __init__(self, cfg: Config, checkpoint_dir: Path | str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.ncfg = cfg.nla
        self.checkpoint_dir = Path(checkpoint_dir)
        self.meta: NLAMeta = load_meta(self.checkpoint_dir)
        if self.meta.role != "av":
            raise ValueError(
                f"{checkpoint_dir} has role={self.meta.role!r}; point this at the "
                f"AV (verbaliser) checkpoint, not the AR"
            )
        if self.meta.injection_scale is None:
            raise ValueError("AV sidecar has no injection_scale - cannot inject")

        self._up = upstream()
        self.device = self.ncfg.device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_dir))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer_report = verify_tokenizer(self.tokenizer, self.meta)
        if not self.tokenizer_report.get("ok"):
            raise RuntimeError(
                f"injection-site verification failed: "
                f"{self.tokenizer_report.get('error')}. Refusing to run - this is "
                f"the failure upstream warns produces silent nonsense."
            )

        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True}
        if self.device != "cpu":
            kwargs["device_map"] = {"": 0}
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.checkpoint_dir), **kwargs
        ).eval()

        self.embed = self.model.get_input_embeddings()
        self.embed_scale = self._up.resolve_embed_scale(self.checkpoint_dir)
        self.d_model = self.model.config.hidden_size
        log.info(
            "AV ready: d_model=%d inj_scale=%.1f embed_scale=%.2f site=%d | %s",
            self.d_model, self.meta.injection_scale, self.embed_scale,
            self.tokenizer_report["position"], loading.cuda_memory(),
        )

    # -- prompt -----------------------------------------------------------

    def _prompt_ids(self):
        import torch

        content = self.meta.av_template.format(injection_char=self.meta.injection_char)
        ids = chat_template_ids(self.tokenizer, content)
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def _build_embeds(self, vectors):
        """``[B, T, d]`` embeddings with each row's activation injected.

        Note that ``embed_tokens`` is an ``nn.Embedding`` and bitsandbytes only
        quantises ``nn.Linear``, so the embedding table is still bf16 here and
        the injected vector enters the network exactly as computed. Only the
        transformer weights are 4-bit.
        """
        import torch

        ids = self._prompt_ids()
        B = vectors.shape[0]
        ids_b = ids.expand(B, -1).contiguous()

        with torch.no_grad():
            emb = (self.embed(ids_b.to(self.embed.weight.device)) * self.embed_scale).float()

        v = torch.as_tensor(np.asarray(vectors, dtype=np.float32))
        if v.ndim == 1:
            v = v.view(1, -1)
        if v.shape[-1] != self.d_model:
            raise ValueError(f"activation width {v.shape[-1]} != d_model {self.d_model}")
        if not torch.isfinite(v).all():
            raise ValueError("activation contains NaN or Inf")

        v_scaled = self._up.normalize_activation(v.float(), self.meta.injection_scale)
        injected = self._up.inject_at_marked_positions(
            ids_b,
            emb.cpu(),
            v_scaled,
            self.meta.injection_token_id,
            self.meta.injection_left_neighbor_id,
            self.meta.injection_right_neighbor_id,
        )
        return injected, ids_b.shape[1]

    # -- generation --------------------------------------------------------

    def verbalise(
        self,
        vectors: np.ndarray,
        seed: int,
        n_samples: int | None = None,
        source: str = "activation",
        batch_size: int = 2,
    ) -> list[list[Verbalisation]]:
        """Verbalise each row of ``vectors``; returns ``n_samples`` per row."""
        import torch

        n_samples = n_samples or self.ncfg.n_samples
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        out: list[list[Verbalisation]] = [[] for _ in range(vectors.shape[0])]

        for s in range(n_samples):
            sample_seed = seed + 1000 * s
            i = 0
            bs = batch_size
            while i < vectors.shape[0]:
                take = min(bs, vectors.shape[0] - i)
                try:
                    texts, ntoks = self._generate(vectors[i : i + take], sample_seed + i)
                except torch.cuda.OutOfMemoryError:
                    loading.free_cuda()
                    if take == 1:
                        raise
                    bs = max(1, take // 2)
                    log.warning("AV OOM - batch size -> %d", bs)
                    continue
                for j, (text, nt) in enumerate(zip(texts, ntoks)):
                    integ = score_integrity(text, self.ncfg.min_ascii_fraction)
                    out[i + j].append(
                        Verbalisation(
                            text=integ["explanation"] or text,
                            explanation=integ["explanation"],
                            raw=text,
                            sample_index=s,
                            seed=sample_seed + i + j,
                            n_tokens=nt,
                            ascii_fraction=integ["ascii_fraction"],
                            cjk_chars=integ["cjk_chars"],
                            well_formed=integ["well_formed"],
                            ok=integ["ok"],
                            source=source,
                        )
                    )
                i += take
        return out

    def _generate(self, vectors: np.ndarray, seed: int) -> tuple[list[str], list[int]]:
        import torch

        embeds, prompt_len = self._build_embeds(vectors)
        embeds = embeds.to(self.model.device, dtype=torch.bfloat16)
        attn = torch.ones(embeds.shape[:2], dtype=torch.long, device=embeds.device)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        with torch.inference_mode():
            seqs = self.model.generate(
                inputs_embeds=embeds,
                attention_mask=attn,
                max_new_tokens=self.ncfg.max_new_tokens,
                do_sample=self.ncfg.temperature > 0,
                temperature=self.ncfg.temperature,
                top_p=self.ncfg.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                # The AV is trained to close with </explanation>. Stopping
                # there rather than running to the cap cuts generation roughly
                # in half, and the tail after the closing tag is discarded by
                # score_integrity anyway.
                stop_strings=["</explanation>"],
                tokenizer=self.tokenizer,
            )
        # With inputs_embeds the model returns only the new tokens - there are
        # no input ids to echo back.
        texts, counts = [], []
        for row in range(seqs.shape[0]):
            ids = seqs[row].tolist()
            counts.append(len(ids))
            texts.append(self.tokenizer.decode(ids, skip_special_tokens=True))
        return texts, counts

    def gaussian_control(self, reference: np.ndarray, rng) -> np.ndarray:
        """Random vectors at the reference rows' L2 norms.

        The Li et al. control: if the verbaliser produces the same claims from
        noise at matched norm as it does from a real activation, those claims
        come from its own priors, not from the target model. Matching the norm
        matters - an off-norm vector would fail injection for a trivial reason
        and the control would be vacuous.
        """
        ref = np.atleast_2d(np.asarray(reference, dtype=np.float32))
        noise = rng.standard_normal(ref.shape).astype(np.float32)
        noise /= np.linalg.norm(noise, axis=1, keepdims=True).clip(1e-12)
        return noise * np.linalg.norm(ref, axis=1, keepdims=True)

    def close(self) -> None:
        try:
            del self.model
            del self.embed
        except AttributeError:
            pass
        gc.collect()
        loading.free_cuda()
