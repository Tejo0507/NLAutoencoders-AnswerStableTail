"""What the readout knows at the menu chunks: an information audit (D81). CPU only.

Implements ``docs/SUPPLEMENTARY_INFORMATION_PLAN.md`` on a run that already
has a registered-outcome O3 analysis. It asks whether the readout's score
(Q1), its text (Q2) and the numbers it names (Q3) carry information about the
model's state, beyond cheaper sources, at design A's menu chunks.

Before it reports anything it checks that every input is byte-identical to
what the registered analysis used, and that the safety labels it rebuilds
are the registered ones.

    python scripts/information_audit.py --config main --run-id main_nla240
    python scripts/information_audit.py --config main
    python scripts/information_audit.py --config replication --forced-dir runs/rep1_forced

Writes ``runs/<run_id>_information/information.json`` and refuses to replace
it (``--force``). Nothing under ``runs/<run_id>/`` is written.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from _stage import base_parser, load_config

from nlaast.analysis.information import (
    Design,
    MentionUnit,
    compare,
    grouped_folds,
    icc1,
    mean_difference,
    mention_contrast,
    numbers_in,
    out_of_fold,
    plain_number,
    problem_draws,
    reading,
    weights_for,
    weighted_auc,
    weighted_log_loss,
    within_problem_concordance,
)
from nlaast.analysis.o3_registered import holm
from nlaast.analysis.position import safe_vector, scenario_answers
from nlaast.baselines.stopping import OUTCOME_FORCED
from nlaast.data.answers import equivalent
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.provenance import git_commit, utcnow
from nlaast.trace.chunking import Chunk, prefix_text

PLAN = Path(__file__).resolve().parents[1] / "docs" / "SUPPLEMENTARY_INFORMATION_PLAN.md"
SEED = 20261020
ROBUST_SEEDS = (20261021, 20261022, 20261023, 20261024, 20261025)
REPS = 2000
N_FOLDS = 10
MARGIN = 0.02
R_CHARS = 600
REAL_KINDS = ("tail", "matched_position", "matched_length")


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_units(cfg, ro, traces, completed, finished, forced_dir):
    """One row per distinct menu chunk, with every feature, text and label."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from position_truncation_audit import cut_off_map
    from score_baselines import build_rows

    log = logging.getLogger("information")
    scores, menu_sets, pids = ro.rule_scores(cfg, log, list(traces))
    registered = {p.problem_id: p for p in ro.problems_for(traces, pids, OUTCOME_FORCED, completed)}
    rows = build_rows([traces[q] for q in pids], OUTCOME_FORCED, completed, "stored")
    cut = cut_off_map(traces, pids, finished)
    tail_start = {r["problem_id"]: r.get("tail_start")
                  for r in read_jsonl(cfg.dir / "ast" / "ast.jsonl")}

    expl: dict[tuple[str, int], list[tuple[str, str]]] = {}
    by_row: dict[tuple[str, int], dict[str, list[tuple[str, str]]]] = {}
    cos_pairs: dict[tuple[str, str, int], dict[int, float]] = {}
    for r in read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl"):
        if r["window_kind"] not in REAL_KINDS or not r.get("integrity_ok"):
            continue
        key = (r["problem_id"], int(r["chunk_index"]))
        expl.setdefault(key, []).append((r["id"], r.get("explanation") or ""))
        row_id = f"{r['problem_id']}|{r['window_kind']}|{r['chunk_index']}"
        by_row.setdefault(key, {}).setdefault(row_id, []).append((r["id"], r.get("explanation") or ""))
        cos_pairs.setdefault((r["problem_id"], r["window_kind"], int(r["chunk_index"])),
                             {})[int(r["sample_index"])] = float(r["cosine"])
    noise: dict[tuple[str, int], list[str]] = {}
    for v in read_jsonl(cfg.dir / "nla" / "verbalisations.jsonl"):
        if v["window_kind"] not in REAL_KINDS:
            continue
        key = (v["problem_id"], int(v["chunk_index"]))
        for s in v.get("noise_samples") or []:
            noise.setdefault(key, []).append(s.get("text") or "")

    units = []
    for pid in pids:
        t = traces[pid]
        pr = registered[pid]
        r = rows[pid]
        co = [cut.get((pid, i)) for i in range(pr.n)]
        commit = safe_vector(scenario_answers(r["answers"], co, r["final_answer"],
                                              "committed_to_final"),
                             r["answer_known"], r["gold"], r["final_answer"], equivalent)
        chunks = [Chunk(**{k: c[k] for k in ("index", "text", "char_start", "char_end",
                                             "token_end", "prefix_tokens")}) for c in t["chunks"]]
        ts = tail_start.get(pid)
        for i in sorted(menu_sets.get(pid, ())):
            prefix = prefix_text(chunks, i, t["trace"])
            texts = [x for _, x in sorted(expl.get((pid, i), []))]
            rows_here = by_row.get((pid, i), {})
            first_row = sorted(rows_here)[0] if rows_here else None
            e2 = [x for _, x in sorted(rows_here[first_row])] if first_row else []
            units.append({
                "E2": "\n".join(e2), "n_rows": len(rows_here), "n_samples": len(texts),
                "problem_id": pid, "chunk": i, "dataset": t.get("dataset"),
                "prefix_tokens": int(t["chunks"][i]["prefix_tokens"]),
                "total_tokens": int(t["n_tokens"]),
                "probe": scores["probe"][pid][i],
                "readout": scores["nla_readout"][pid][i],
                "Y_reg": bool(pr.safe[i]), "Y_commit": bool(commit[i]),
                "Y_tail": None if ts is None else bool(i >= ts),
                "cut_off": co[i],
                "E": "\n".join(texts), "R": prefix[-R_CHARS:],
                "N": "\n".join(noise.get((pid, i), [])),
                "question": t["question"], "prefix": prefix,
                "final_answer": t.get("final_answer"),
            })
    pairs = [(v[0], v[1]) for v in cos_pairs.values() if 0 in v and 1 in v]
    return units, pids, pairs


def numeric(units, blocks):
    cols = []
    for u in units:
        row = []
        if "pos" in blocks:
            row += [np.log(u["prefix_tokens"]), u["chunk"], float(u["dataset"] == "math")]
        if "probe" in blocks:
            p = u["probe"]
            row += [0.0 if p is None else float(_logit(p)), float(p is None)]
        if "readout" in blocks:
            row += [float(u["readout"])]
        if "diag" in blocks:
            row += [u["prefix_tokens"] / u["total_tokens"], np.log(u["total_tokens"])]
        cols.append(row)
    return np.asarray(cols, float)


TEXT_BLOCKS = {"E": "E", "R": "R", "N": "N", "E2": "E2", "Rfull": "prefix"}


def design(units, spec):
    num_blocks = [b for b in spec.split("+") if b in ("pos", "probe", "readout", "diag")]
    text_blocks = tuple([u[TEXT_BLOCKS[b]] for u in units] for b in spec.split("+")
                        if b in TEXT_BLOCKS)
    return Design(numeric(units, num_blocks), text_blocks)


POST_HOC_CONTRASTS = {
    "E2 over pos+R (multiplicity-safe Q2-P)": ("pos+R+E2", "pos+R"),
    "E over pos+Rfull": ("pos+Rfull+E", "pos+Rfull"),
    "A over pos": ("pos+A", "pos"),
    "E over pos+A": ("pos+A+E", "pos+A"),
    "E over pos+R+A": ("pos+R+A+E", "pos+R+A"),
    "pos+E minus pos+A (head to head)": ("pos+E", "pos+A"),
    "E over pos (reference)": ("pos+E", "pos"),
    "ensemble(A,E) minus pos+A (deviation 2)": ("ens(A,E)", "pos+A"),
}


def activations_for(units, acts_dir, layer):
    """The layer's activation at each unit's chunk boundary: the verbaliser's input."""
    from nlaast.activations.store import ActivationStore

    store = ActivationStore(acts_dir)
    cache = {}
    X = []
    for u in units:
        pid = u["problem_id"]
        if pid not in cache:
            arr, meta = store.get(pid, layer)
            idx = meta.get("chunk_indices", list(range(arr.shape[0])))
            cache[pid] = (arr, {int(c): r for r, c in enumerate(idx)})
        arr, rowof = cache[pid]
        X.append(np.asarray(arr[rowof[u["chunk"]]], np.float32))
    return np.vstack(X)


def post_hoc(units, keys, counts, X, progress, labels=("Y_reg", "Y_tail"), terms=True):
    """Plan deviation 1: multiplicity, the whole prefix, and the same-label probe."""
    from nlaast.analysis.information import nested_probe

    out = {"n_units_with_more_than_one_row": int(sum(u["n_rows"] > 1 for u in units)),
           "samples_per_unit": {str(k): int(sum(u["n_samples"] == k for u in units))
                                for k in sorted({u["n_samples"] for u in units})}}
    for lab in labels:
        keep = [k for k, u in enumerate(units) if u[lab] is not None]
        sub = [units[k] for k in keep]
        y = np.array([u[lab] for u in sub], bool)
        groups = np.array([u["problem_id"] for u in sub])
        folds = grouped_folds(y, groups, N_FOLDS, SEED)
        progress(f"{lab}: nested activation probe")
        extras, Cs = nested_probe(X[keep], y, groups, folds, seed=SEED)
        oof = {}
        for spec in ("pos", "pos+R", "pos+E", "pos+R+E2", "pos+Rfull", "pos+Rfull+E",
                     "pos+A", "pos+A+E", "pos+R+A", "pos+R+A+E"):
            progress(f"{lab} {spec}")
            base_spec = "+".join(b for b in spec.split("+") if b != "A")
            oof[spec] = out_of_fold(design(sub, base_spec), y, folds,
                                    extras if "A" in spec.split("+") else None)
        # deviation 2: an unfitted equal-weight logit average
        la, le = _logit(oof["pos+A"]), _logit(oof["pos+E"])
        oof["ens(A,E)"] = 1 / (1 + np.exp(-(la + le) / 2))
        res = {"n_units": int(len(y)), "chosen_C": Cs,
               "auc": {m: weighted_auc(y, p) for m, p in oof.items()},
               "contrasts": contrasts(y, groups, oof, POST_HOC_CONTRASTS, keys, counts)}
        res["subgroups_E_over_pos+A"] = subgroup(sub, y, groups, oof, "pos+A+E", "pos+A", keys, counts)
        out[lab] = res
    if terms:
        out["top_E_terms_Y_tail"] = top_terms(units, "Y_tail")
    return out


_TOKEN = __import__("re").compile(r"(?u)\b\w+\b")


def lexicon_score(text: str, positive: list[str], negative: list[str]) -> int:
    """Positive terms present minus negative terms present (deviation 3).
    Unigrams and bigrams, after the TF-IDF tokenisation."""
    toks = _TOKEN.findall((text or "").lower())
    grams = set(toks) | {f"{a} {b}" for a, b in zip(toks, toks[1:])}
    return sum(t in grams for t in positive) - sum(t in grams for t in negative)


def auc_with_ci(y, s, groups, keys, counts):
    W = weights_for(groups, keys, counts)
    rep = np.array([weighted_auc(y, s, w) for w in W])
    ok = rep[np.isfinite(rep)]
    return {"auc": weighted_auc(y, s), "ci95": [float(np.quantile(ok, 0.025)),
                                                float(np.quantile(ok, 0.975))]}


def lexicon_test(units, lex, keys, counts, progress):
    """Deviation 3: the frozen lexicon's transfer, fitted on nothing here."""
    pos_t = [t for t, _ in lex["positive"]]
    neg_t = [t for t, _ in lex["negative"]]
    for u in units:
        u["lex"] = float(lexicon_score(u["E"], pos_t, neg_t))
    out = {}
    for lab in ("Y_tail", "Y_reg"):
        sub = [u for u in units if u[lab] is not None]
        y = np.array([u[lab] for u in sub], bool)
        groups = np.array([u["problem_id"] for u in sub])
        s = np.array([u["lex"] for u in sub])
        res = {"n_units": len(sub), "lexicon_score": auc_with_ci(y, s, groups, keys, counts),
               "prefix_tokens_only": auc_with_ci(y, np.array([u["prefix_tokens"] for u in sub], float),
                                                 groups, keys, counts)}
        folds = grouped_folds(y, groups, N_FOLDS, SEED)
        progress(f"lexicon {lab}")
        pos = numeric(sub, ["pos"])
        p0 = out_of_fold(Design(pos), y, folds)
        p1 = out_of_fold(Design(np.c_[pos, s]), y, folds)
        res["pos+lex minus pos"] = compare(y, p1, p0, weights_for(groups, keys, counts))
        out[lab] = res
    return out


def top_terms(units, lab, k=20):
    """Descriptive: the E terms with the largest weights in a model fitted on all units."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    sub = [u for u in units if u[lab] is not None]
    y = np.array([u[lab] for u in sub], bool)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, lowercase=True,
                          token_pattern=r"(?u)\b\w+\b")
    Xt = vec.fit_transform([u["E"] for u in sub])
    clf = LogisticRegression(C=1.0, max_iter=5000, solver="liblinear").fit(Xt, y)
    w = clf.coef_[0]
    names = vec.get_feature_names_out()
    order = np.argsort(w)
    return {"positive": [[str(names[i]), float(w[i])] for i in order[::-1][:k]],
            "negative": [[str(names[i]), float(w[i])] for i in order[:k]]}


Q1_MODELS = ("pos", "pos+probe", "pos+readout", "pos+probe+readout",
             "pos+diag+probe", "pos+diag+probe+readout")
Q1_CONTRASTS = {
    "Q1-P readout over pos+probe": ("pos+probe+readout", "pos+probe"),
    "readout over pos": ("pos+readout", "pos"),
    "probe over pos": ("pos+probe", "pos"),
    "readout over pos+diag+probe": ("pos+diag+probe+readout", "pos+diag+probe"),
}
Q2_MODELS = ("pos", "pos+R", "pos+E", "pos+R+E", "pos+N")
Q2_CONTRASTS = {
    "Q2-P E over pos+R": ("pos+R+E", "pos+R"),
    "E over pos": ("pos+E", "pos"),
    "R over pos": ("pos+R", "pos"),
    "N over pos (null check)": ("pos+N", "pos"),
}


def run_models(units, ylab, models, seed, progress=None):
    keep = [k for k, u in enumerate(units) if u[ylab] is not None]
    sub = [units[k] for k in keep]
    y = np.array([u[ylab] for u in sub], bool)
    groups = np.array([u["problem_id"] for u in sub])
    folds = grouped_folds(y, groups, N_FOLDS, seed)
    oof = {}
    for m in models:
        if progress:
            progress(f"{ylab} {m} seed {seed}")
        oof[m] = out_of_fold(design(sub, m), y, folds)
    return sub, y, groups, oof


def contrasts(y, groups, oof, table, keys, counts):
    W = weights_for(groups, keys, counts)
    out = {}
    for name, (a, b) in table.items():
        if a in oof and b in oof:
            out[name] = compare(y, oof[a], oof[b], W)
    return out


def subgroup(units_sub, y, groups, oof, a, b, keys, counts):
    ds = np.array([u["dataset"] for u in units_sub])
    res = {}
    reps = {}
    for d in ("gsm8k", "math"):
        m = ds == d
        if m.sum() < 10 or len(np.unique(y[m])) < 2:
            res[d] = None
            continue
        W = weights_for(groups[m], keys, counts)
        res[d] = compare(y[m], oof[a][m], oof[b][m], W)
        res[d]["n_units"] = int(m.sum())
        reps[d] = np.array([weighted_auc(y[m], oof[a][m], w) - weighted_auc(y[m], oof[b][m], w)
                            for w in W])
    if len(reps) == 2:
        diff = reps["gsm8k"] - reps["math"]
        ok = diff[np.isfinite(diff)]
        res["gsm8k_minus_math"] = {
            "delta": res["gsm8k"]["delta_auc"] - res["math"]["delta_auc"],
            "ci95": [float(np.quantile(ok, 0.025)), float(np.quantile(ok, 0.975))]}
    return res


def q3(units, keys, counts, have_noise):
    answers = {}
    for u in units:
        a = plain_number(u["final_answer"])
        if a is not None:
            answers[u["problem_id"]] = (u["dataset"], a)
    elig, mu, mu_noise = [], [], []
    for u in units:
        a = plain_number(u["final_answer"])
        if a is None:
            continue
        ctx = numbers_in(u["question"]) | numbers_in(u["prefix"])
        if a in ctx:
            continue
        elig.append(u)
        mu.append(MentionUnit(u["problem_id"], u["dataset"], a, ctx, numbers_in(u["E"])))
        mu_noise.append(MentionUnit(u["problem_id"], u["dataset"], a, ctx, numbers_in(u["N"])))
    own, other = mention_contrast(mu, answers)
    ok = np.isfinite(other)
    groups = np.array([u["problem_id"] for u in elig])
    out = {"n_units_with_plain_number_answer": sum(plain_number(u["final_answer"]) is not None
                                                   for u in units),
           "n_eligible_units": len(elig), "n_eligible_problems": int(len(set(groups))),
           "own_rate": float(own.mean()) if len(own) else None,
           "other_rate": float(np.nanmean(other)) if ok.any() else None}
    if not ok.any():
        return out, None
    W = weights_for(groups[ok], keys, counts)
    out["Q3-P own minus other"] = mean_difference((own - other)[ok], W)
    pre = np.array([u["Y_tail"] is False for u in elig])
    tail = np.array([u["Y_tail"] is True for u in elig])
    for name, m in (("pre_tail", pre & ok), ("tail", tail & ok)):
        if m.sum():
            out[f"{name} own minus other"] = mean_difference(
                (own - other)[m], weights_for(groups[m], keys, counts))
            out[f"{name} own_rate"] = float(own[m].mean())
    ds = np.array([u["dataset"] for u in elig])
    for d in ("gsm8k", "math"):
        m = (ds == d) & ok
        if m.sum():
            out[f"{d} own minus other"] = mean_difference(
                (own - other)[m], weights_for(groups[m], keys, counts))
    if have_noise:
        own_n, _ = mention_contrast(mu_noise, answers)
        out["noise_own_rate"] = float(own_n.mean())
        out["own minus noise_own"] = mean_difference(own - own_n, weights_for(groups, keys, counts))
    yreg = np.array([u["Y_reg"] for u in elig], bool)
    ment = own.astype(bool)
    if ment.any() and (~ment).any():
        W2 = weights_for(groups, keys, counts)
        rep = []
        for w in W2:
            a1, a0 = w[ment].sum(), w[~ment].sum()
            rep.append(((w[ment] @ yreg[ment]) / a1 - (w[~ment] @ yreg[~ment]) / a0)
                       if a1 > 0 and a0 > 0 else np.nan)
        rep = np.array(rep)
        rok = rep[np.isfinite(rep)]
        out["P(Y_reg | mention) - P(Y_reg | none)"] = {
            "with_mention": float(yreg[ment].mean()), "n_mention": int(ment.sum()),
            "without": float(yreg[~ment].mean()), "n_without": int((~ment).sum()),
            "difference": float(yreg[ment].mean() - yreg[~ment].mean()),
            "ci95": [float(np.quantile(rok, 0.025)), float(np.quantile(rok, 0.975))]}
    return out, out["Q3-P own minus other"]


def main() -> int:
    p = base_parser(__doc__)
    p.add_argument("--forced-dir", default=None, help="completion output (default: runs/main_forced)")
    p.add_argument("--reps", type=int, default=REPS)
    p.add_argument("--skip-robustness", action="store_true",
                   help="omit the CV-seed robustness refits (plan §3)")
    p.add_argument("--post-hoc", action="store_true",
                   help="plan deviation 1 only: writes information_posthoc.json")
    p.add_argument("--acts-dir", default=None,
                   help="activation store (default: the run's acts/); must hold these traces")
    p.add_argument("--freeze-lexicon", action="store_true",
                   help="deviation 3: copy this run's top E terms to lexicon.json and exit")
    p.add_argument("--lexicon", default=None, metavar="PATH",
                   help="deviation 3: score this run with a frozen lexicon.json")
    p.add_argument("--extra-label", default=None, metavar="PATH",
                   help="D82 §4: labels_O256.jsonl; Q1 and the same-label probe under it")
    args = p.parse_args()
    cfg = load_config(args)
    info_dir = cfg.dir.parent / f"{cfg.run_id}_information"
    if args.freeze_lexicon:
        src = info_dir / "information_posthoc.json"
        dst = info_dir / "lexicon.json"
        if dst.exists():
            print(f"{dst} exists; a frozen lexicon is never replaced", file=sys.stderr)
            return 1
        import hashlib
        terms = read_json(src)["top_E_terms_Y_tail"]
        write_json(dst, {"source": str(src),
                         "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                         "frozen": utcnow(), "positive": terms["positive"],
                         "negative": terms["negative"]})
        print("wrote", dst)
        return 0
    out_path = info_dir / ("information_O256.json" if args.extra_label else
                           "information_lexicon.json" if args.lexicon else
                           "information_posthoc.json" if args.post_hoc else "information.json")
    if out_path.exists() and not args.force:
        print(f"{out_path} exists; refusing to overwrite (use --force)", file=sys.stderr)
        return 1
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import reanalyse_o3_registered as ro

    logging.basicConfig(level=logging.ERROR)
    stored_path = cfg.dir.parent / f"{cfg.run_id}_o3_registered" / "o3_registered.json"
    if not stored_path.exists():
        print(f"{stored_path} is missing: the registered analysis must exist first", file=sys.stderr)
        return 2
    stored = read_json(stored_path)
    forced_dir = Path(args.forced_dir) if args.forced_dir else cfg.dir.parent / "main_forced"
    traces_path = cfg.dir / "traces" / "traces.jsonl"
    paths = {"traces": traces_path, "forced": forced_dir / "forced.jsonl",
             "reconstructions": cfg.dir / "nla" / "reconstructions.jsonl",
             "boundary_scores": cfg.dir / "baselines" / "boundary_scores.json"}
    for k, path in paths.items():
        if stored["inputs_sha256"][k] != ro.sha256(path):
            print(f"{k} changed since the registered analysis", file=sys.stderr)
            return 2
    traces = {t["problem_id"]: t for t in read_jsonl(traces_path) if t.get("ok")}
    _, completed, _, finished, agree, _ = ro.load_forced(forced_dir, ro.sha256(traces_path))
    if agree is None or agree < ro.MIN_AGREEMENT:
        print(f"forcing instrument reproduced on {agree}", file=sys.stderr)
        return 2

    t0 = time.monotonic()
    def progress(msg):
        print(f"  [{time.monotonic() - t0:6.0f}s] {msg}", flush=True)

    units, pids, cos_pairs = build_units(cfg, ro, traces, completed, finished, forced_dir)
    # the labels must be the registered ones
    reg = {p.problem_id: p for p in ro.problems_for(traces, pids, OUTCOME_FORCED, completed)}
    for u in units:
        if u["Y_reg"] != bool(reg[u["problem_id"]].safe[u["chunk"]]):
            raise SystemExit("rebuilt safety labels differ from the registered ones")
    have_noise = any(u["N"] for u in units)
    keys, counts = problem_draws(pids, args.reps, np.random.default_rng(SEED))

    if args.extra_label:
        import hashlib
        lab_path = Path(args.extra_label)
        lab = {(r["problem_id"], int(r["chunk"])): bool(r["Y_O256"]) for r in read_jsonl(lab_path)}
        missing = [u for u in units if (u["problem_id"], u["chunk"]) not in lab]
        if missing:
            raise SystemExit(f"{len(missing)} units have no O256 label in {lab_path}")
        for u in units:
            u["Y_ext"] = lab[(u["problem_id"], u["chunk"])]
        progress("Q1 under O256")
        sub, y, groups, oof = run_models(units, "Y_ext", Q1_MODELS, SEED, progress)
        acts_dir = Path(args.acts_dir) if args.acts_dir else cfg.dir / "acts"
        X = activations_for(units, acts_dir, cfg.target.layer)
        out = {"status": "D82 §4: the D81 information audit under the 256-token outcome",
               "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(),
               "labels": str(lab_path),
               "labels_sha256": hashlib.sha256(lab_path.read_bytes()).hexdigest(),
               "inputs_sha256": {k: ro.sha256(v) for k, v in paths.items()},
               "n_labels_changed_vs_Y_reg": int(sum(u["Y_ext"] != u["Y_reg"] for u in units)),
               "Y_ext": {"n_units": int(len(y)), "positive_rate": float(y.mean()),
                         "auc": {m: weighted_auc(y, oof[m]) for m in oof},
                         "Q1": contrasts(y, groups, oof, Q1_CONTRASTS, keys, counts)},
               "probe_and_text": post_hoc(units, keys, counts, X, progress, labels=("Y_ext",),
                                          terms=False)["Y_ext"]}
        write_json(out_path, out)
        print(json.dumps({k: [round(v["delta_auc"], 4), v["ci95"]]
                          for k, v in out["Y_ext"]["Q1"].items()}, indent=1))
        print("wrote", out_path)
        return 0

    if args.lexicon:
        import hashlib
        lex_path = Path(args.lexicon)
        lex = read_json(lex_path)
        # the lexicon's own problems: those of the run it was learned on
        src_run = Path(lex["source"]).parent.name.removesuffix("_information")
        src_pids = {r["problem_id"] for r in read_jsonl(
            cfg.dir.parent / src_run / "nla" / "reconstructions.jsonl")}
        overlap = len(set(pids) & src_pids)
        out = {"status": ("NOT evidence: in sample, the lexicon was learned on "
                          f"{overlap} of these problems (code check only)" if overlap else
                          "pre-specified secondary (D81 deviation 3); out of sample"),
               "n_problems_shared_with_lexicon_source": overlap,
               "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(),
               "lexicon": str(lex_path),
               "lexicon_sha256": hashlib.sha256(lex_path.read_bytes()).hexdigest(),
               "inputs_sha256": {k: ro.sha256(v) for k, v in paths.items()},
               **lexicon_test(units, lex, keys, counts, progress)}
        write_json(out_path, out)
        print(json.dumps({k: out[k] for k in ("Y_tail", "Y_reg")}, indent=1, default=str)[:2500])
        print("wrote", out_path)
        return 0

    if args.post_hoc:
        acts_dir = Path(args.acts_dir) if args.acts_dir else cfg.dir / "acts"
        acts_traces = acts_dir.parent / "traces" / "traces.jsonl"
        if ro.sha256(acts_traces) != ro.sha256(traces_path):
            print(f"{acts_dir} was extracted from other traces", file=sys.stderr)
            return 2
        X = activations_for(units, acts_dir, cfg.target.layer)
        out = {"status": ("specified before any rep1 outcome; secondary (D81 deviation 1)"
                          if cfg.run_id == "rep1" else
                          "post hoc (D81 deviation 1): added after Q2-P was seen"),
               "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(),
               "plan_sha256": ro.sha256(PLAN), "seed": SEED, "reps": args.reps,
               "layer": cfg.target.layer, "acts_dir": str(acts_dir),
               "inputs_sha256": {k: ro.sha256(v) for k, v in paths.items()},
               **post_hoc(units, keys, counts, X, progress)}
        out["elapsed_s"] = time.monotonic() - t0
        if out_path.exists():
            # a regeneration must reproduce every value it shares with the old file
            old = read_json(out_path)
            for lab in ("Y_reg", "Y_tail"):
                for k, v in old[lab]["contrasts"].items():
                    n = out[lab]["contrasts"][k]
                    if abs(n["delta_auc"] - v["delta_auc"]) > 1e-9 or n["ci95"] != v["ci95"]:
                        raise SystemExit(f"{lab} {k} did not reproduce; not overwriting")
            out["reproduced_previous"] = True
        write_json(out_path, out)
        print(json.dumps({lab: {k: [round(v["delta_auc"], 4), [round(x, 4) for x in v["ci95"]],
                                    v["p_bootstrap"]] for k, v in out[lab]["contrasts"].items()}
                          for lab in ("Y_reg", "Y_tail")}, indent=1))
        print("wrote", out_path)
        return 0

    out = {
        "status": ("pre-specified secondary to D75 (D81)" if cfg.run_id == "rep1" else
                   "exploratory (D81): this run's other results were seen before the plan"),
        "run_id": cfg.run_id, "created": utcnow(), "git": git_commit(),
        "plan_sha256": ro.sha256(PLAN), "seed": SEED, "reps": args.reps, "n_folds": N_FOLDS,
        "inputs_sha256": {k: ro.sha256(v) for k, v in paths.items()},
        "n_problems": len(pids), "n_units": len(units),
        "units_by_dataset": {d: sum(u["dataset"] == d for u in units) for d in ("gsm8k", "math")},
        "label_rates": {lab: float(np.mean([u[lab] for u in units if u[lab] is not None]))
                        for lab in ("Y_reg", "Y_tail", "Y_commit")},
        "probe_missing_rate": float(np.mean([u["probe"] is None for u in units])),
        "noise_text_available": have_noise,
    }

    q2_models = Q2_MODELS if have_noise else tuple(m for m in Q2_MODELS if "N" not in m)
    primaries = {}
    for lab in ("Y_reg", "Y_tail", "Y_commit"):
        models = Q1_MODELS + (q2_models if lab != "Y_commit" else ())
        models = tuple(dict.fromkeys(models))
        sub, y, groups, oof = run_models(units, lab, models, SEED, progress)
        res = {"n_units": int(len(y)), "positive_rate": float(y.mean()),
               "auc": {m: weighted_auc(y, oof[m]) for m in oof},
               "log_loss": {m: weighted_log_loss(y, oof[m]) for m in oof},
               "within_problem_concordance": {m: within_problem_concordance(y, oof[m], groups)
                                              for m in oof},
               "Q1": contrasts(y, groups, oof, Q1_CONTRASTS, keys, counts)}
        res["within_problem_concordance"]["raw prefix_tokens"] = within_problem_concordance(
            y, [u["prefix_tokens"] for u in sub], groups)
        res["within_problem_concordance"]["raw readout"] = within_problem_concordance(
            y, [u["readout"] for u in sub], groups)
        if lab != "Y_commit":
            res["Q2"] = contrasts(y, groups, oof, Q2_CONTRASTS, keys, counts)
        if lab == "Y_reg":
            primaries["Q1-P"] = res["Q1"]["Q1-P readout over pos+probe"]
            primaries["Q2-P"] = res["Q2"]["Q2-P E over pos+R"]
            res["Q4_subgroups"] = {
                "Q1-P": subgroup(sub, y, groups, oof, "pos+probe+readout", "pos+probe", keys, counts),
                "Q2-P": subgroup(sub, y, groups, oof, "pos+R+E", "pos+R", keys, counts)}
            if not args.skip_robustness:
                rob = {"Q1-P": [], "Q2-P": []}
                for s in ROBUST_SEEDS:
                    _, ys, gs, o = run_models(units, lab, ("pos+probe", "pos+probe+readout",
                                                           "pos+R", "pos+R+E"), s, progress)
                    rob["Q1-P"].append(weighted_auc(ys, o["pos+probe+readout"]) - weighted_auc(ys, o["pos+probe"]))
                    rob["Q2-P"].append(weighted_auc(ys, o["pos+R+E"]) - weighted_auc(ys, o["pos+R"]))
                res["cv_seed_robustness"] = {k: {"seeds": list(ROBUST_SEEDS), "delta_auc": v,
                                                 "min": min(v), "max": max(v)} for k, v in rob.items()}
        out[lab] = res

    q3res, q3p = q3(units, keys, counts, have_noise)
    out["Q3"] = q3res
    if q3p is not None:
        primaries["Q3-P"] = {"delta": q3p["mean"], "ci95": q3p["ci95"], "p_bootstrap": q3p["p_bootstrap"]}

    # Q5: reliability of the readout score
    if cos_pairs:
        a, b = np.array(cos_pairs).T
        out["Q5"] = {"n_pairs": len(cos_pairs), "pearson_s0_s1": float(pearsonr(a, b)[0]),
                     "spearman_s0_s1": float(spearmanr(a, b)[0]),
                     "icc1_readout_by_problem": icc1([u["readout"] for u in units],
                                                     [u["problem_id"] for u in units])}

    pv = {k: v["p_bootstrap"] for k, v in primaries.items()}
    adj = holm(pv)
    out["primaries"] = {}
    for k, v in primaries.items():
        delta = v.get("delta_auc", v.get("delta"))
        out["primaries"][k] = {"delta": delta, "ci95": v["ci95"], "p": v["p_bootstrap"],
                               "p_holm": adj[k]["p_holm"],
                               "reading": reading(v["ci95"], adj[k]["reject"], delta, MARGIN)}
    out["elapsed_s"] = time.monotonic() - t0
    write_json(out_path, out)
    print(json.dumps({"primaries": out["primaries"], "n_units": len(units)}, indent=1, default=str))
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
