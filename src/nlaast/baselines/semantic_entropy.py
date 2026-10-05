"""Semantic entropy (Farquhar et al.), adapted to mathematical answers.

The original samples several generations, clusters them by *bidirectional NLI
entailment*, and takes the entropy over meaning clusters rather than over token
sequences.

**Adaptation, stated plainly.** For mathematical answers the correct equivalence
relation is symbolic equality, not textual entailment: ``\\frac{1}{2}``, ``0.5``
and ``1/2`` are one meaning, and no off-the-shelf NLI model reliably says so.
Clustering therefore uses the same external verifier the rest of the study uses
(``data/answers.py``). This keeps the clustering consistent with how correctness
and answer stability are defined everywhere else - using NLI here would mean the
entropy baseline and the tail detector disagreed about what "the same answer"
means, and no comparison between them would be interpretable.

The consequence is recorded as a deviation in PROJECT_PLAN.md section 3 and in
the execution report. This is not "semantic entropy" as published; it is
semantic entropy with a task-appropriate equivalence relation, and it is
labelled ``semantic_entropy_symbolic`` in all outputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any, Sequence

import numpy as np

from ..data.answers import cluster_answers

METHOD_NAME = "semantic_entropy_symbolic"


@dataclass
class EntropyResult:
    n_samples: int
    n_parsed: int
    n_clusters: int
    #: Entropy over meaning clusters, in nats.
    semantic_entropy: float
    #: Entropy over raw answer strings - the lexical comparator.
    lexical_entropy: float
    #: Share of samples in the largest cluster; the simplest consistency signal.
    top_cluster_share: float
    #: Share agreeing with the trace's own final answer.
    agreement_with_final: float
    clusters: list[dict[str, Any]] = field(default_factory=list)
    method: str = METHOD_NAME

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log(c / total) for c in counts if c > 0))


def semantic_entropy(
    answers: Sequence[str | None],
    final_answer: str | None = None,
) -> EntropyResult:
    """Cluster sampled answers and take the entropy over clusters.

    Unparseable samples become singleton clusters rather than being dropped.
    Dropping them would make a problem on which the model mostly fails to
    produce an answer look *confident*, which inverts the signal exactly where
    it matters most.
    """
    answers = list(answers)
    n = len(answers)
    if n == 0:
        return EntropyResult(0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"))

    clusters = cluster_answers(answers)
    counts = [len(c.members) for c in clusters]
    sem = _entropy(counts)

    lex_counts: dict[str, int] = {}
    for a in answers:
        key = "<none>" if a is None else str(a).strip()
        lex_counts[key] = lex_counts.get(key, 0) + 1
    lex = _entropy(list(lex_counts.values()))

    agreement = float("nan")
    if final_answer is not None:
        from ..data.answers import equivalent

        agreement = float(np.mean([equivalent(a, final_answer) for a in answers]))

    return EntropyResult(
        n_samples=n,
        n_parsed=sum(1 for a in answers if a is not None),
        n_clusters=len(clusters),
        semantic_entropy=sem,
        lexical_entropy=lex,
        top_cluster_share=max(counts) / n,
        agreement_with_final=agreement,
        clusters=[
            {"representative": c.representative, "n": len(c.members),
             "members": list(c.members)}
            for c in clusters
        ],
    )


def normalised_entropy(result: EntropyResult) -> float:
    """Entropy divided by its maximum for this sample count.

    Needed because the number of samples per boundary is not always equal once
    generation failures are counted, and raw nats are not comparable across
    different sample counts.
    """
    if result.n_samples <= 1:
        return 0.0
    return result.semantic_entropy / math.log(result.n_samples)


def confidence_score(result: EntropyResult) -> float:
    """A stopping score in [0, 1]: high means confident, so stop.

    Expressed as confidence rather than entropy so every rule in
    ``stopping.py`` fires on a *high* score and the sweep code does not need
    per-rule polarity flags.
    """
    return 1.0 - normalised_entropy(result)
