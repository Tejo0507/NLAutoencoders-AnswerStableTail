"""Position-only control and forced-answer truncation for design A (D79).

Implements the computations of ``docs/POSITION_TRUNCATION_PLAN.md``. Pure
numpy and scipy; no I/O. The thresholded-rule machinery is
``analysis.o3_registered``'s, unchanged, so every new rule is scored exactly
as the registered ones are.

Why a position control is needed: design A's menu is symmetric about the tail
start by construction, so within it later chunks are nearly always safe. Any
score that rises with position can then beat the no-signal rule, which favours
early chunks. These rules ask how much of the readout's design-A result
position alone reproduces.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .o3_registered import Problem, RuleMatrix, bootstrap_p, gap_over_budgets

#: Position scores (plan §A.2). ``relative`` uses the trace's final length,
#: which no online rule has, so it is a diagnostic of the menu's geometry and
#: never a fair control. ``tokens_falling`` (plan deviation 3) is ``tokens``
#: negated: the same information, ranked the other way.
POSITION_KINDS = ("tokens", "index", "relative", "tokens_falling")
SCENARIOS = ("registered", "uncommitted", "committed_to_final")


def position_scores(prefix_tokens: Sequence[int], total_tokens: int, kind: str) -> list[float]:
    """A score per boundary that depends on nothing but where the boundary is."""
    if kind not in POSITION_KINDS:
        raise ValueError(f"kind must be one of {POSITION_KINDS}, not {kind!r}")
    if kind == "tokens":
        return [float(t) for t in prefix_tokens]
    if kind == "tokens_falling":
        return [-float(t) for t in prefix_tokens]
    if kind == "index":
        return [float(i) for i in range(len(prefix_tokens))]
    if total_tokens <= 0:
        raise ValueError("relative position needs a positive total length")
    return [float(t) / total_tokens for t in prefix_tokens]


def _menu_points(scores, prefix_tokens, menu) -> tuple[list[int], list[float], list[float]]:
    idx, x, y = [], [], []
    for i, allowed in enumerate(menu):
        s = scores[i]
        if not allowed or s is None or not np.isfinite(s):
            continue
        if prefix_tokens[i] <= 0:
            raise ValueError(f"boundary {i} has {prefix_tokens[i]} prefix tokens; log undefined")
        idx.append(i)
        x.append(float(np.log(prefix_tokens[i])))
        y.append(float(s))
    return idx, x, y


def residualise_on_position(
    scores: Mapping[str, Sequence[float | None]],
    prefix_tokens: Mapping[str, Sequence[int]],
    menu: Mapping[str, np.ndarray],
) -> dict[str, list[float | None]]:
    """Each menu score minus its leave-one-problem-out OLS prediction from
    ``[1, log prefix_tokens]`` (plan §A.2).

    The fit for problem *p* uses the menu chunks of every other problem, so a
    problem's own scores never set its own trend. No outcome is used. Off-menu
    and unscored boundaries are ``None``: design A cannot stop there.
    """
    pts = {p: _menu_points(scores[p], prefix_tokens[p], np.asarray(menu[p], bool))
           for p in scores}
    xtx, xty = np.zeros((2, 2)), np.zeros(2)
    per: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for p, (_, x, y) in pts.items():
        X = np.column_stack([np.ones(len(x)), x]) if x else np.zeros((0, 2))
        Y = np.asarray(y, float)
        a, b = X.T @ X, X.T @ Y
        per[p] = (a, b)
        xtx += a
        xty += b
    out: dict[str, list[float | None]] = {}
    for p, (idx, x, y) in pts.items():
        a, b = per[p]
        A = xtx - a
        if np.linalg.matrix_rank(A) < 2:
            raise ValueError(f"{p}: the other problems cannot identify a position trend")
        beta = np.linalg.solve(A, xty - b)
        res: list[float | None] = [None] * len(scores[p])
        for i, xi, yi in zip(idx, x, y):
            res[i] = yi - (beta[0] + beta[1] * xi)
        out[p] = res
    return out


def within_problem_spearman(
    scores: Mapping[str, Sequence[float | None]],
    positions: Mapping[str, Sequence[float]],
    menu: Mapping[str, np.ndarray],
    min_points: int = 3,
) -> dict[str, Any]:
    """Spearman correlation of a score with position over each problem's menu.

    Problems with fewer than ``min_points`` scored menu chunks, or a constant
    score, are skipped and counted. Outcome-free (plan §A.3).
    """
    rhos, skipped = [], 0
    for p in scores:
        m = np.asarray(menu[p], bool)
        pairs = [(float(positions[p][i]), float(scores[p][i])) for i in range(len(m))
                 if m[i] and scores[p][i] is not None and np.isfinite(scores[p][i])]
        if len(pairs) < min_points or len({s for _, s in pairs}) < 2:
            skipped += 1
            continue
        rho = spearmanr([a for a, _ in pairs], [b for _, b in pairs]).statistic
        rhos.append(float(rho))
    r = np.asarray(rhos)
    return {"n_problems": int(len(r)), "n_skipped": skipped,
            "mean_rho": float(r.mean()) if len(r) else None,
            "median_rho": float(np.median(r)) if len(r) else None,
            "share_positive": float((r > 0).mean()) if len(r) else None}


def paired_comparisons(
    n: int,
    matrices: Mapping[str, RuleMatrix],
    no_signal: tuple[np.ndarray, np.ndarray] | None,
    pairs: Mapping[str, tuple[str, str]],
    budgets: Sequence[float],
    tol: float,
    reps: int,
    rng: np.random.Generator,
    min_budgets: int,
) -> dict[str, dict[str, Any]]:
    """Every pair's mean safe-rate gap, on one shared paired problem bootstrap.

    Each replicate draws problem weights exactly as
    ``reanalyse_o3_registered.analyse`` does, and as its only use of ``rng``.
    So the same seed reproduces the registered design-A intervals, and every
    pair here is computed on the same resamples. ``no_signal`` is the closed
    form ``(ES, EQ)`` per problem; it is the rule named ``"no_signal"``.
    """
    def curves(w):
        c = {r: m.curve(w) for r, m in matrices.items()}
        if no_signal is not None:
            ES, EQ = no_signal
            if w is None:
                c["no_signal"] = (ES.mean(0), EQ.mean(0))
            else:
                ww = w / w.sum()
                c["no_signal"] = (ww @ ES, ww @ EQ)
        return c

    full = curves(None)
    out: dict[str, dict[str, Any]] = {}
    for name, (a, b) in pairs.items():
        g, k, diffs = gap_over_budgets(full[a], full[b], budgets, tol)
        out[name] = {"gap": g, "n_budgets_matched": k, "per_budget": diffs}
    boot = {name: np.full(reps, np.nan) for name in pairs}
    for i in range(reps):
        w = rng.multinomial(n, np.full(n, 1 / n)).astype(float)
        c = curves(w)
        for name, (a, b) in pairs.items():
            g, k, _ = gap_over_budgets(c[a], c[b], budgets, tol)
            if k >= min_budgets:
                boot[name][i] = g
    for name in pairs:
        r = boot[name]
        ok = r[np.isfinite(r)]
        evaluable = out[name]["n_budgets_matched"] >= min_budgets
        out[name].update({
            "evaluable": evaluable,
            "ci95": ([float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))]
                     if evaluable and len(ok) else None),
            "p_bootstrap": bootstrap_p(r, 1 / reps) if evaluable else float("nan"),
            "n_replicates_evaluable": int(len(ok)),
        })
    return out


def position_reading(gaps: Mapping[str, Mapping[str, Any]], holm_x: Mapping[str, Mapping[str, Any]],
                     margin: float = 0.05) -> dict[str, Any]:
    """Plan §A.4, applied mechanically to family X."""
    def sig_pos(k):
        return bool(holm_x[k]["reject"]) and gaps[k]["gap"] > 0

    x1, x2, x4 = "readout - position_tokens", "position_tokens - no_signal", \
        "readout_residualised - no_signal"
    beyond = sig_pos(x1) and sig_pos(x4)
    ci = gaps[x1]["ci95"]
    excluded = bool(ci is not None and ci[1] < margin)
    if beyond:
        overall = "the readout carries stopping information beyond position"
    elif excluded:
        overall = f"an advantage of {margin} or more over position is excluded"
    else:
        overall = "unresolved"
    return {"beyond_position": beyond, "advantage_over_position_excluded": excluded,
            "position_beats_no_signal": sig_pos(x2),
            "beats_no_signal_is_not_evidence_beyond_position": sig_pos(x2),
            "overall": overall}


def falling_reading(gaps: Mapping[str, Mapping[str, Any]], holm_y: Mapping[str, Mapping[str, Any]]
                    ) -> dict[str, Any]:
    """Plan deviation 3, family Y, applied mechanically."""
    y1, y2 = "readout - position_tokens_falling", "position_tokens_falling - no_signal"
    ci = gaps[y1]["ci95"]
    y2_pos = bool(holm_y[y2]["reject"]) and gaps[y2]["gap"] > 0
    y1_pos = bool(holm_y[y1]["reject"]) and gaps[y1]["gap"] > 0
    reproduces = y2_pos and ci is not None and ci[0] <= 0 <= ci[1]
    overall = ("a falling-position rule reproduces the readout's advantage over no signal"
               if reproduces else
               "the readout beats the falling-position rule" if y1_pos else "unresolved")
    return {"falling_beats_no_signal": y2_pos, "readout_beats_falling": y1_pos,
            "overall": overall}


# ---- Part B: truncation ---------------------------------------------------

def scenario_answers(
    answers: Sequence[str | None],
    cut_off: Sequence[bool | None],
    final_answer: str | None,
    scenario: str,
) -> list[str | None]:
    """Forced answers with cut-off completions read one way (plan §B.4).

    Only boundaries whose completion is known to be cut off (``True``) change.
    ``None`` (status unknown) and ``False`` keep the measured answer.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, not {scenario!r}")
    if len(answers) != len(cut_off):
        raise ValueError("answers and cut-off flags differ in length")
    if scenario == "registered":
        return list(answers)
    repl = None if scenario == "uncommitted" else final_answer
    return [repl if c is True else a for a, c in zip(answers, cut_off)]


def safe_vector(
    answers: Sequence[str | None],
    known: Sequence[bool],
    gold: str | None,
    final_answer: str | None,
    equivalent: Callable[[Any, Any], bool],
) -> np.ndarray:
    """Registered safety (plan D69 §2): a stop is safe iff the forced answer's
    correctness equals the final answer's. Unknown boundaries are not safe
    here; ``Problem.known`` keeps a rule from being scored at them."""
    final_ok = equivalent(final_answer, gold)
    return np.array([bool(k) and (equivalent(a, gold) == final_ok)
                     for a, k in zip(answers, known)])


def relabelled(problem: Problem, safe: np.ndarray) -> Problem:
    return Problem(problem.problem_id, problem.saving, np.asarray(safe, bool), problem.known)


def majority_answer(answers: Sequence[str | None], equivalent: Callable[[Any, Any], bool]
                    ) -> str | None:
    """An answer that more than half of ``answers`` are equivalent to, else ``None``.

    Unparsed continuations count towards the denominator, not towards any
    answer (plan §B.2).
    """
    n = len(answers)
    for a in answers:
        if a is None:
            continue
        if sum(1 for b in answers if b is not None and equivalent(a, b)) * 2 > n:
            return a
    return None


def stop_cut_off_share(
    matrix: RuleMatrix,
    columns: Sequence[int | None],
    problems: Sequence[Problem],
    cut_off: Mapping[tuple[str, int], bool | None],
) -> list[dict[str, Any] | None]:
    """At each matched column: the share of the rule's stops on a cut-off completion.

    The denominator is the stops whose cut-off status is known; stops with an
    unknown status are counted separately, never as either.
    """
    out: list[dict[str, Any] | None] = []
    for k in columns:
        if k is None:
            out.append(None)
            continue
        cut = known = unknown = 0
        for p, b in zip(problems, matrix.boundary[:, k]):
            if b < 0:
                continue
            c = cut_off.get((p.problem_id, int(b)))
            if c is None:
                unknown += 1
            else:
                known += 1
                cut += bool(c)
        out.append({"n_stops": known + unknown, "n_status_unknown": unknown,
                    "share_cut_off": cut / known if known else None})
    return out


def rate(records: Sequence[Mapping[str, Any]], flag: str, by: str | None = None
         ) -> dict[str, Any]:
    """Share of ``records`` with ``flag`` true, overall or per value of ``by``.

    Records whose flag is ``None`` are excluded from that share and counted.
    """
    groups: dict[str, list[Mapping[str, Any]]] = {"all": list(records)}
    if by is not None:
        groups = {}
        for r in records:
            groups.setdefault(str(r.get(by)), []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        vals = [r[flag] for r in rs if r.get(flag) is not None]
        out[g] = {"n": len(rs), "n_unknown": len(rs) - len(vals),
                  "rate": (sum(bool(v) for v in vals) / len(vals)) if vals else None}
    return out
