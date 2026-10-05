"""Answer-Stable Tail detection (O1 / RQ1).

Deliberately free of any dependency on activations or on the autoencoder. The
review's first distinction - answer stability is not causal redundancy - is
enforced by this package's import graph, not by a comment.
"""

from .detector import (
    ASTResult,
    BoundaryEvidence,
    TailStatus,
    convergence_boundary,
    detect_ast,
    matched_windows,
)

__all__ = [
    "ASTResult",
    "BoundaryEvidence",
    "TailStatus",
    "convergence_boundary",
    "detect_ast",
    "matched_windows",
]
