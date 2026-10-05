"""Typed configuration.

Every experiment setting lives here and reaches the code as a frozen dataclass.
No model name, layer index, seed, sample count or path is written anywhere else.

Configs are YAML files under ``configs/``. ``base.yaml`` holds the defaults; an
overlay (``pilot.yaml``, ``main.yaml``, ...) is deep-merged over it, and CLI
``--set a.b=c`` assignments are merged last. The fully resolved mapping is
hashed and that hash goes into the run manifest, so a result can always be
traced to the exact settings that produced it.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from . import paths


# --- Sections ---


@dataclass(frozen=True)
class TargetModelConfig:
    """The model under study. Fixed by the NLA checkpoint, not a free choice.

    The released autoencoder is trained against ``Qwen2.5-7B-Instruct`` at layer
    20 and is not transferable (Project_Review_II.md section 2.1.8), so
    ``repo_id`` and ``layer`` are coupled to ``nla.av_repo``.
    """

    repo_id: str = "Qwen/Qwen2.5-7B-Instruct"
    revision: str | None = None
    layer: int = 20
    d_model: int = 3584
    #: ``bf16`` | ``nf4``. ``nf4`` is forced by 6.44 GB of VRAM; see PROJECT_PLAN.md.
    precision: str = "nf4"
    device: str = "cuda"
    max_position_embeddings: int = 8192


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding for the target model's reasoning traces."""

    max_new_tokens: int = 640
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    #: Greedy pass used for the canonical trace; sampling is for the controls.
    canonical_greedy: bool = True
    batch_size: int = 4
    system_prompt: str = (
        "You are a careful mathematical reasoner. Think step by step, then give "
        "the final answer on its own last line in the form 'The answer is X.'"
    )
    #: Forces an answer out of a truncated prefix without letting it reason further.
    force_answer_suffix: str = "\n\nTherefore, the answer is"
    force_answer_max_new_tokens: int = 24


@dataclass(frozen=True)
class DataConfig:
    datasets: tuple[str, ...] = ("gsm8k", "math")
    #: MATH subjects staged locally; the full benchmark has seven.
    math_subjects: tuple[str, ...] = ("algebra", "counting_and_probability")
    n_gsm8k: int = 150
    n_math: int = 100
    split: str = "test"
    #: Fraction of problems reserved for fitting probes and tail directions.
    train_fraction: float = 0.4
    seed: int = 20261005


@dataclass(frozen=True)
class ChunkConfig:
    """Sentence-level segmentation of a trace (Liu & Wang's construction)."""

    min_chunk_chars: int = 12
    max_chunks: int = 40
    #: A second, stricter segmentation used by falsification test F4/F5.
    alt_mode: str = "newline"


@dataclass(frozen=True)
class ASTConfig:
    """Answer-Stable Tail detection. No activation input - this is L1/O1."""

    #: Independently sampled continuations per candidate boundary.
    k_continuations: int = 5
    continuation_temperature: float = 0.8
    continuation_max_new_tokens: int = 200
    #: Agreement-window width for the answer-convergence *baseline* (Liu & Wang).
    convergence_window: int = 2
    #: All continuations must agree with the full-trace answer to accept a boundary.
    require_unanimous: bool = True
    #: Candidate boundaries are searched from this fraction of the trace onward.
    min_boundary_fraction: float = 0.0
    #: Cap on how many boundaries the backwards scan evaluates per trace.
    #:
    #: Criterion 3 requires every boundary at or after the tail start to
    #: qualify, so the scan runs from the end. Capping it means a tail longer
    #: than this is reported as exactly this long. The bias is one-directional:
    #: it *under*-states tail length, never overstates it, which is the safe
    #: direction for any claim about how much of a trace is redundant. The cap
    #: is recorded per problem so affected traces are identifiable.
    max_boundaries_evaluated: int = 12
    seed: int = 7


@dataclass(frozen=True)
class NLAConfig:
    """Released autoencoder. Conventions come from each checkpoint's nla_meta.yaml."""

    av_repo: str = "kitft/nla-qwen2.5-7b-L20-av"
    ar_repo: str = "kitft/nla-qwen2.5-7b-L20-ar"
    precision: str = "nf4"
    device: str = "cuda"
    #: Verbalisation decoding. The AV emits a few hundred tokens per activation.
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.95
    #: Samples per activation; >1 supports the resampling arm of the audit.
    n_samples: int = 2
    #: Windows verbalised per problem, spread over the tail.
    windows_per_problem: int = 3
    #: Also verbalise the matched-position and matched-length control windows.
    verbalise_controls: bool = True
    #: Cap on problems entering the NLA arm. The verbaliser emits hundreds of
    #: tokens per activation, so this arm is an order of magnitude more
    #: expensive than the behavioural ones; its N is allowed to be smaller, and
    #: the achieved N is reported separately. 0 means no cap.
    max_problems: int = 0
    seed: int = 1234
    #: Gate thresholds (PROJECT_PLAN.md section 2, Q2).
    min_ascii_fraction: float = 0.75
    reference_fve: float = 0.752


@dataclass(frozen=True)
class ProbeConfig:
    """Hidden-state correctness probe (Zhang et al.) - a baseline, not a mechanism."""

    kind: str = "logistic"
    C: float = 1.0
    max_iter: int = 2000
    standardise: bool = True
    #: Splits are grouped by problem id; a problem never straddles the split.
    n_folds: int = 5
    seed: int = 11


@dataclass(frozen=True)
class SemanticEntropyConfig:
    """Farquhar et al., adapted: meaning clusters are symbolic-equivalence classes.

    The original clusters free-form text by bidirectional NLI entailment. For
    mathematical answers the correct equivalence relation is symbolic equality
    (``\\frac{1}{2}`` and ``0.5`` are the same answer and no NLI model reliably
    says so), so clustering uses the external verifier. Documented as an
    adaptation in PROJECT_PLAN.md section 3.
    """

    n_samples: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 640
    seed: int = 99


@dataclass(frozen=True)
class FaithfulnessConfig:
    """Claim-level audit - O4/RQ2. Reconstruction change is the score, not the verdict."""

    max_claims_per_explanation: int = 8
    #: Paraphrase arm supplies the null distribution for the deletion threshold.
    n_paraphrases: int = 3
    null_percentile: float = 95.0
    #: Li et al. control: AV run on a Gaussian vector at matched L2 norm.
    verbaliser_only_control: bool = True
    n_verbaliser_only: int = 1
    seed: int = 555


@dataclass(frozen=True)
class CausalConfig:
    """O5/RQ3. Direction claims need matched-random-direction *and* matched-position."""

    interventions: tuple[str, ...] = (
        "truncate",
        "filler",
        "ablate_dir",
        "random_dir",
        "matched_position",
    )
    #: Dose-response sweep over projection coefficient (Zhang & Nanda).
    dose_coefficients: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
    n_random_directions: int = 5
    filler_text: str = "Let us consider this further. "
    max_problems: int = 60
    seed: int = 31337


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap_iterations: int = 10000
    confidence: float = 0.95
    #: Benjamini-Hochberg across the pre-registered family (F9).
    fdr_q: float = 0.05
    seed: int = 2024


@dataclass(frozen=True)
class RobustnessConfig:
    """Falsification battery - PROJECT_PLAN.md section 6."""

    enabled_tests: tuple[str, ...] = (
        "F1_length",
        "F2_difficulty",
        "F3_verbaliser_only",
        "F4_parser",
        "F5_ast_sensitivity",
        "F6_probe_leakage",
        "F7_layer_sweep",
        "F8_quantisation",
        "F10_position",
    )
    ast_sweep_k: tuple[int, ...] = (3, 5)
    ast_sweep_temperature: tuple[float, ...] = (0.6, 0.8, 1.0)
    layer_sweep: tuple[int, ...] = (14, 20, 24)
    quantisation_subsample: int = 24


@dataclass(frozen=True)
class Config:
    run_id: str = "dev"
    description: str = ""
    seed: int = 20261005
    stages: tuple[str, ...] = (
        "env",
        "data",
        "models",
        "traces",
        "acts",
        "ast",
        "baselines",
        "nla",
        "faithfulness",
        "causal",
        "analysis",
        "robustness",
        "report",
    )
    target: TargetModelConfig = field(default_factory=TargetModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    ast: ASTConfig = field(default_factory=ASTConfig)
    nla: NLAConfig = field(default_factory=NLAConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    semantic_entropy: SemanticEntropyConfig = field(default_factory=SemanticEntropyConfig)
    faithfulness: FaithfulnessConfig = field(default_factory=FaithfulnessConfig)
    causal: CausalConfig = field(default_factory=CausalConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

    # -- derived ----------------------------------------------------------

    @property
    def dir(self) -> Path:
        return paths.run_dir(self.run_id)

    def stage_dir(self, stage: str) -> Path:
        d = self.dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --- Loading ---

_SECTIONS: dict[str, type] = {
    "target": TargetModelConfig,
    "generation": GenerationConfig,
    "data": DataConfig,
    "chunk": ChunkConfig,
    "ast": ASTConfig,
    "nla": NLAConfig,
    "probe": ProbeConfig,
    "semantic_entropy": SemanticEntropyConfig,
    "faithfulness": FaithfulnessConfig,
    "causal": CausalConfig,
    "analysis": AnalysisConfig,
    "robustness": RobustnessConfig,
}


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(cls: type, value: Any) -> Any:
    """Build a dataclass from a mapping, converting lists to tuples.

    Tuples rather than lists because ``Config`` is frozen and hashed; a mutable
    default would let a stage quietly change a setting mid-run.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping for {cls.__name__}, got {type(value).__name__}")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(value) - set(known)
    if unknown:
        raise KeyError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, raw in value.items():
        ftype = known[name].type
        if isinstance(raw, list):
            kwargs[name] = tuple(raw)
        elif isinstance(ftype, str) and ftype.startswith("tuple") and isinstance(raw, tuple):
            kwargs[name] = raw
        else:
            kwargs[name] = raw
    return cls(**kwargs)


def from_mapping(mapping: Mapping[str, Any]) -> Config:
    mapping = dict(mapping)
    kwargs: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        if key in mapping:
            kwargs[key] = _coerce(cls, mapping.pop(key))
    for key in ("run_id", "description", "seed"):
        if key in mapping:
            kwargs[key] = mapping.pop(key)
    if "stages" in mapping:
        kwargs["stages"] = tuple(mapping.pop("stages"))
    if mapping:
        raise KeyError(f"unknown top-level config keys: {sorted(mapping)}")
    return Config(**kwargs)


def parse_overrides(assignments: Sequence[str]) -> dict[str, Any]:
    """``['nla.n_samples=4', 'data.n_gsm8k=20']`` -> nested dict.

    Values go through the YAML scalar parser, so ``4`` is an int, ``0.5`` a
    float, ``true`` a bool and ``[1,2]`` a list, matching the YAML files.
    """
    out: dict[str, Any] = {}
    for item in assignments:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        key, raw = item.split("=", 1)
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load(
    overlay: str | Path | None = None,
    overrides: Sequence[str] = (),
    base: str | Path | None = None,
) -> Config:
    """Resolve base.yaml + overlay + CLI overrides into a frozen ``Config``."""
    base_path = Path(base) if base else paths.CONFIGS / "base.yaml"
    merged: dict[str, Any] = {}
    if base_path.exists():
        merged = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if overlay is not None:
        op = Path(overlay)
        if not op.exists():
            op = paths.CONFIGS / f"{overlay}.yaml"
        if not op.exists():
            raise FileNotFoundError(f"no config overlay {overlay!r} (looked at {op})")
        merged = deep_merge(merged, yaml.safe_load(op.read_text(encoding="utf-8")) or {})
    if overrides:
        merged = deep_merge(merged, parse_overrides(overrides))
    return from_mapping(merged)


def save(cfg: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=True, allow_unicode=True),
        encoding="utf-8",
    )
