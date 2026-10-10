"""Stage `report`: figures, tables and the run report.

Reads only saved stage outputs, so it reruns in seconds without touching a
model. It reports what the run actually produced: a stage that did not run is
listed as not run, not omitted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from _stage import base_parser, setup, should_skip

from nlaast.analysis import tables
from nlaast.logging_utils import read_json, read_jsonl, write_json
from nlaast.viz import figures

STAGE = "report"


def main() -> int:
    args = base_parser(__doc__).parse_args()
    cfg, manifest, log = setup(args)
    if should_skip(manifest, STAGE, args.force, log):
        return 0
    manifest.start_stage(STAGE)

    out_dir = cfg.stage_dir(STAGE)
    fig_dir = out_dir / "figures"
    tbl_dir = out_dir / "tables"

    ast_rows = read_jsonl(cfg.dir / "ast" / "ast.jsonl")
    recon = read_jsonl(cfg.dir / "nla" / "reconstructions.jsonl")
    audits = read_jsonl(cfg.dir / "faithfulness" / "audits.jsonl")
    curves_path = cfg.dir / "analysis" / "tables" / "stopping_at_matched_budget.json"
    results_path = cfg.dir / "analysis" / "results.json"
    results = read_json(results_path) if results_path.exists() else {}
    robust_path = cfg.dir / "robustness" / "robustness.json"
    robust = read_json(robust_path) if robust_path.exists() else {}

    made: dict[str, str | None] = {}
    if ast_rows:
        made["tail_distribution"] = figures.fig_tail_distribution(ast_rows, fig_dir)
    bl_curves = cfg.dir / "baselines" / "curves.json"
    if bl_curves.exists():
        raw = read_json(bl_curves).get("curves", {})
        made["stopping_curves"] = figures.fig_stopping_curves(raw, fig_dir)
    if recon:
        made["reconstruction_by_window"] = figures.fig_reconstruction_by_window(
            recon, fig_dir)
    if audits:
        made["faithfulness"] = figures.fig_faithfulness(audits, fig_dir)
    causal_path = cfg.dir / "causal" / "summary.json"
    if causal_path.exists():
        made["dose_response"] = figures.fig_dose_response(read_json(causal_path), fig_dir)
    if robust.get("F8_quantisation"):
        made["quantisation_drift"] = figures.fig_quantisation_drift(
            robust["F8_quantisation"], fig_dir)

    # Copy the analysis tables next to the figures so the report directory is
    # self-contained.
    src_tbl = cfg.dir / "analysis" / "tables"
    if src_tbl.exists():
        tbl_dir.mkdir(parents=True, exist_ok=True)
        for f in src_tbl.glob("*"):
            (tbl_dir / f.name).write_bytes(f.read_bytes())

    md = build_markdown(cfg, manifest, results, robust, ast_rows, recon, audits, made)
    (out_dir / "RESULTS.md").write_text(md, encoding="utf-8")
    write_json(out_dir / "manifest_snapshot.json", manifest.data)

    log.info("figures: %s", {k: bool(v) for k, v in made.items()})
    log.info("report written to %s", out_dir / "RESULTS.md")
    manifest.finish_stage(STAGE, status="complete", output=str(out_dir),
                          metrics={"figures": {k: bool(v) for k, v in made.items()}})
    return 0


def _fmt(v, nd=3):
    if v is None:
        return "not available"
    if isinstance(v, float):
        return "not available" if v != v else f"{v:.{nd}f}"
    return str(v)


def build_markdown(cfg, manifest, results, robust, ast_rows, recon, audits, figs) -> str:
    env = manifest.data.get("environment", {})
    stages = manifest.data.get("stages", {})

    L: list[str] = []
    A = L.append
    A(f"# Results - run `{cfg.run_id}`")
    A("")
    A(f"Config hash `{cfg.hash()}` | commit "
      f"`{(manifest.data.get('git') or {}).get('commit', 'unknown')}` | "
      f"generated {manifest.data.get('updated')}")
    A("")
    A("Produced entirely from saved stage outputs by `scripts/write_run_report.py`. "
      "Stages that did not run are listed as such rather than omitted.")
    A("")

    A("## Environment")
    A("")
    A(f"- {env.get('gpu_name', 'no GPU')}, "
      f"{env.get('gpu_vram_gb', '?')} GB VRAM, "
      f"{env.get('ram_total_gb', '?')} GB RAM")
    A(f"- torch {env.get('torch')} / CUDA {env.get('cuda_version')}, "
      f"transformers {env.get('transformers')}, bitsandbytes {env.get('bitsandbytes')}")
    A(f"- Python {str(env.get('python', '')).split()[0]} on {env.get('platform')}")
    A("")

    A("## Stage status")
    A("")
    A("| stage | status | elapsed (s) |")
    A("|---|---|---|")
    for s in cfg.stages:
        e = stages.get(s, {})
        A(f"| {s} | {e.get('status', 'not run')} | {_fmt(e.get('elapsed_s'), 0)} |")
    A("")

    # ---------------------------------------------------------------- RQ1
    A("## RQ1 - the Answer-Stable Tail")
    A("")
    rq1 = results.get("rq1") or {}
    if rq1:
        A(f"Detected on **{rq1.get('n_with_tail', 0)} of {rq1.get('n', 0)}** problems "
          f"({_fmt(rq1.get('tail_rate'))}), of which "
          f"{rq1.get('n_trivial_tail', 0)} cover only the final chunk - a tail "
          f"that short means no redundancy was detected. Non-trivial tail rate: "
          f"**{_fmt(rq1.get('nontrivial_tail_rate'))}**.")
        A("")
        A(f"**Read `tail_rate` with care.** At the final boundary the prefix is "
          f"the whole trace, so forcing an answer from it reproduces the final "
          f"answer and continuations from it restate the same answer: both "
          f"criteria are close to tautological there. The final boundary "
          f"qualifies on {_fmt(rq1.get('final_boundary_qualifies_rate'))} of "
          f"problems here, which puts `tail_rate` near its ceiling by "
          f"construction. The quantity that carries information is how *much* "
          f"of the trace the tail covers.")
        A("")
        if rq1.get("n_with_tail"):
            A(f"- tail fraction: mean **{_fmt(rq1.get('mean_tail_fraction'))}**, "
              f"median {_fmt(rq1.get('median_tail_fraction'))}, range "
              f"{_fmt(rq1.get('min_tail_fraction'), 2)}-"
              f"{_fmt(rq1.get('max_tail_fraction'), 2)} of chunks")
            A(f"- tail tokens as a share of all generated tokens: "
              f"**{_fmt(rq1.get('tail_token_share'))}**")
            A(f"- mean tail length: {_fmt(rq1.get('mean_tail_tokens'), 1)} tokens")
        A("")
        fab = rq1.get("forced_answer_budget")
        sweep_path = cfg.dir / "ast" / "sweep_F5.json"
        relaxed = (read_json(sweep_path).get("forced_budget_relaxed")
                   if sweep_path.exists() else None)
        if fab and fab.get("n_cut_off_and_blocking"):
            A("### The forcing budget bounds the tail from below")
            A("")
            A(f"Criterion 1 forces an answer out of a truncated prefix under "
              f"`generation.force_answer_max_new_tokens` "
              f"({cfg.generation.force_answer_max_new_tokens} tokens) so the "
              f"model commits rather than starting a fresh derivation. When it "
              f"starts writing one out anyway, the budget cuts it off and the "
              f"extractor reads whatever number happens to be last. That "
              f"happened on **{_fmt(fab['cut_off_rate'])}** of evaluated "
              f"boundaries, and on **{_fmt(fab['blocking_rate'])}** it was "
              f"*blocking*: the forced answer was truncated and failed "
              f"criterion 1 while every resampled continuation from the same "
              f"prefix reached the final answer. "
              f"{fab['n_problems_affected']} problems are affected.")
            A("")
            A("Such a boundary fails for a budget reason, not an evidential "
              "one, and it moves the tail start later - so the detected tail "
              "is a **lower** bound.")
            A("")
            if relaxed and relaxed.get("mean_tail_fraction") is not None:
                A(f"Counting those boundaries as satisfying criterion 1 gives "
                  f"the upper end of the interval. Mean tail fraction: "
                  f"**{_fmt(rq1.get('mean_tail_fraction'))}** as detected, "
                  f"**{_fmt(relaxed.get('mean_tail_fraction'))}** relaxed. "
                  f"Tail share of generated tokens: "
                  f"**{_fmt(rq1.get('tail_token_share'))}** to "
                  f"**{_fmt(relaxed.get('tail_token_share'))}**.")
                A("")
                A("The relaxed figure is not the pre-registered criterion and "
                  "is reported only as the other end of the interval. Every "
                  "number elsewhere in this report uses the detected tail.")
                A("")

        dvs = rq1.get("determinacy_vs_statement")
        if dvs:
            A("### Determinacy is not statement")
            A("")
            A(f"The criteria ask whether the answer is *determined* from a "
              f"prefix - forced and resampled answers from it all match the "
              f"final answer. They do not ask whether the trace has yet "
              f"**said** the answer. On this corpus the tail begins before the "
              f"answer is stated on **{_fmt(dvs['tail_starts_before_answer_stated'])}** "
              f"of problems, by a mean of {_fmt(dvs['mean_chunks_before_statement'], 2)} "
              f"chunks and up to {dvs['max_chunks_before_statement']}.")
            A("")
            A("Two consequences, in opposite directions:")
            A("")
            A("- **For the stopping comparison this is exactly right.** A rule "
              "that stops where the answer is already determined loses nothing, "
              "and the tokens it skips are genuinely saved.")
            A("- **For reading the window as post-answer behaviour it is not.** "
              "A window that begins before the answer is stated contains "
              "computation that produces the answer, not redundant "
              "verification of it. Verbalisations sampled from early in such a "
              "tail should not be read as descriptions of a model checking its "
              "work, and the faithfulness and causal sections below are "
              "qualified by that.")
            A("")
            if dvs.get("n_unstated"):
                A(f"On {dvs['n_unstated']} problems the strict extractor never "
                  f"found the final answer stated anywhere in the trace.")
                A("")

        A("Detection status, edge cases retained:")
        A("")
        A(tables.markdown(tables.ast_status_table(ast_rows)) if ast_rows else "_none_")
        A("")
        cva = rq1.get("convergence_vs_ast")
        if cva:
            A(f"The cheap agreement rule fires **earlier** than the AST on "
              f"{_fmt(cva['convergence_earlier'])} of problems, at the same point on "
              f"{_fmt(cva['same'])}, and later on {_fmt(cva['convergence_later'])} "
              f"(mean gap {_fmt(cva['mean_gap_chunks'], 2)} chunks). This is the "
              f"quantitative form of the concern Mo et al. raise about consensus "
              f"stopping, measured on this run's own traces.")
            A("")
    else:
        A("_Stage did not run._")
        A("")

    # ------------------------------------------------------- primary RQ
    A("## Primary research question - does the verbalised readout add value?")
    A("")
    cmp_path = Path(cfg.dir / "analysis" / "tables" / "stopping_at_matched_budget.csv")
    if cmp_path.exists():
        df = pd.read_csv(cmp_path)
        A("Every rule read off its own safety/saving curve at matched token budget. "
          "Comparing rules at their own operating points would compare different "
          "budgets, under which a rule that merely stops later always looks safer.")
        A("")
        A(tables.markdown(df))
        A("")
    else:
        A("_Stopping comparison did not run._")
        A("")
    # How each comparison signal was produced. At these sample sizes this is
    # not a footnote: a baseline scored in-sample, or an arm that is absent
    # rather than merely weak, changes how the table above should be read.
    bl = stages.get("baselines", {}).get("metrics") or {}
    probe_path = cfg.dir / "baselines" / "probe.json"
    probe_meta = read_json(probe_path) if probe_path.exists() else {}
    if bl or probe_meta:
        A("### How each signal was scored")
        A("")
        A(f"- rules on the curve: {', '.join(bl.get('rules', [])) or 'none'}")
        if probe_meta:
            tsf = probe_meta.get("train_split_fit") or {}
            A(f"- correctness probe: per-boundary scores are "
              f"`{probe_meta.get('probe_source')}` over "
              f"{probe_meta.get('n_boundaries_scored')} boundaries, so no problem "
              f"is scored by a probe that saw it. Grouped-CV AUC "
              f"{_fmt((probe_meta.get('report') or {}).get('auc'))}"
              + (f"; train-split fit held out AUC {_fmt(tsf.get('eval_auc'))} on "
                 f"{tsf.get('n_eval')} boundaries." if tsf.get("available")
                 else "; no train-split fit was possible at this corpus size."))
            floor = probe_meta.get("positional_floor") or {}
            if floor.get("auc") is not None:
                above = floor.get("probe_auc_above_floor")
                A(f"- **what the probe is reading.** The same model family and "
                  f"folds, fitted on activation-free positional features "
                  f"(`{'`, `'.join(floor.get('features', []))}`), reach AUC "
                  f"{_fmt(floor['auc'])}. The probe is "
                  f"{_fmt(above)} AUC above that floor.")
                if above is not None and above < 0.05:
                    A("")
                    A("  The probe's label — *the intermediate answer at this "
                      "boundary is already correct* — becomes true once the "
                      "trace has worked the answer out and stays true, so it "
                      "is strongly ordered by position. At this margin the "
                      "layer-20 probe is largely reading **where in the trace "
                      "a boundary sits**, not whether the answer is correct. "
                      "Read its row in the comparison above as a positional "
                      "stopping rule with a small activation-derived "
                      "increment, not as a correctness readout.")
        if "semantic_entropy_available" in bl:
            if bl["semantic_entropy_available"]:
                A(f"- semantic entropy: scored on "
                  f"{bl.get('n_problems_with_entropy_score')} problems "
                  f"(symbolic-equivalence clustering, not NLI - see DECISIONS.md D3)")
            else:
                A("- semantic entropy: **unavailable** on every problem, so the arm "
                  "is absent from the table above rather than reported as weak")
        A("")

    nla_info = results.get("nla_stopping") or {}
    if nla_info and not nla_info.get("available", True):
        A(f"The NLA arm could not be placed on the curve: {nla_info.get('reason')}.")
        A("")
    elif nla_info.get("available"):
        A(f"The NLA score is {nla_info.get('score')}. Reducing a free-text "
          f"description to one scalar is a choice, and a different reduction could "
          f"give a different curve; this study does not claim to have found the "
          f"best one.")
        A("")

    # ---------------------------------------------------------------- RQ2
    A("## RQ2 - claim-level faithfulness")
    A("")
    head = results.get("rq2_headline") or {}
    if head:
        A(f"- claims audited: **{head['n_claims']}**")
        A(f"- reconstruction-dependent: **{_fmt(head['reconstruction_dependent'])}**")
        A(f"- reproduced from a Gaussian vector at matched norm (verbaliser-only "
          f"control): **{_fmt(head['reproduced_from_noise'])}**")
        A(f"- dependent **and** not reproduced from noise: "
          f"**{_fmt(head['dependent_and_not_noise'])}**")
        A("")
        A("The last line is the one that matters. A claim that carries "
          "reconstruction weight but is also produced from noise is attributable "
          "to the verbaliser's priors rather than to the target activation "
          "(Li et al.).")
        A("")
        A("This is a **RECAP-inspired** evaluation. It adopts RECAP's principle of "
          "independent verification; it does not reproduce RECAP, which co-trains "
          "the target model and cannot be applied retrospectively to a released "
          "checkpoint.")
        A("")
    else:
        A("_Faithfulness audit did not run._")
        A("")

    A(_verbalisation_examples(cfg, n=2))

    if recon:
        A("### Reconstruction fidelity by window")
        A("")
        A("Fidelity is **not** faithfulness; the two are reported separately "
          "throughout, and this table speaks only to the former.")
        A("")
        A(tables.markdown(tables.reconstruction_table(recon)))
        A("")

    # ---------------------------------------------------------------- RQ3
    A("## RQ3 - causal evidence")
    A("")
    rq3 = results.get("rq3") or {}
    if rq3:
        v = rq3.get("verdict", {})
        A(f"**Direction claim supported: {v.get('supported')}**")
        A("")
        A("| requirement | met |")
        A("|---|---|")
        for k in ("beats_random_direction", "beats_matched_position",
                  "monotone_dose_response", "answers_preserved"):
            A(f"| {k.replace('_', ' ')} | {v.get(k)} |")
        A("")
        arms = rq3.get("arms") or {}
        if arms:
            A(tables.markdown(pd.DataFrame(list(arms.values()))))
            A("")
        A("Following Zhang & Nanda, a direction claim requires the candidate to "
          "beat both a matched-random direction and the same direction applied at "
          "a matched pre-stabilisation position, with a monotone dose-response. "
          "These are enforced in code, not asserted in prose.")
        A("")
    else:
        A("_Causal stage did not run. The review scopes it as an extension "
          "contingent on available compute._")
        A("")

    # --------------------------------------------------------- robustness
    A("## Falsification battery")
    A("")
    if robust:
        A("| test | status | key finding |")
        A("|---|---|---|")
        for name, r in sorted(robust.items()):
            A(f"| {name} | {r.get('status')} | {_summarise_test(name, r)} |")
        A("")
    else:
        A("_Robustness stage did not run._")
        A("")

    # ----------------------------------------------------------- testing
    t = results.get("tests") or {}
    if t:
        A("## Hypothesis tests")
        A("")
        A(f"{t['n_tests']} tests in the pre-registered family, {t['n_evaluated']} "
          f"evaluable, {t['n_significant']} significant after Benjamini-Hochberg at "
          f"q = {results.get('fdr_q')}. Tests that could not be evaluated remain in "
          f"the family; removing them would make the correction look kinder than "
          f"it is.")
        A("")
        A(tables.markdown(tables.tests_table(
            [type("T", (), {"to_dict": lambda s, d=d: d})() for d in t["tests"]])))
        A("")

    A("## Figures")
    A("")
    for name, path in figs.items():
        if path:
            A(f"- `{Path(path).name}` - {name.replace('_', ' ')}")
    A("")
    return "\n".join(L)


def _clip(text: str, n: int = 380) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _verbalisation_examples(cfg, n: int = 2) -> str:
    """A few verbalisations verbatim, each beside its own Gaussian control.

    The aggregate faithfulness numbers say what fraction of claims survive the
    controls. They do not let a reader see *what kind* of claim the verbaliser
    makes, and on this corpus that is the most informative thing about it: the
    descriptions identify the register and the discourse position of the window
    accurately while inventing the specifics, and two samples of the same
    activation invent different ones. Printing the text is the only way a
    reader can check that characterisation rather than take it on trust.
    """
    path = cfg.dir / "nla" / "verbalisations.jsonl"
    rows = read_jsonl(path)
    if not rows:
        return ""
    traces = {t["problem_id"]: t for t in
              read_jsonl(cfg.dir / "traces" / "traces.jsonl") if t.get("ok")}

    picked = [r for r in rows if r.get("window_kind") == "tail" and r.get("n_ok")][:n]
    if not picked:
        return ""

    out = ["### Sample verbalisations, with their controls", "",
           "Verbatim, truncated. Each block is one activation window: the "
           "verbaliser's own samples, then the same prompt driven by a "
           "Gaussian vector at matched L2 norm (the Li et al. control). The "
           "gold answer is given so the reader can check the specifics "
           "against it.", ""]
    for r in picked:
        t = traces.get(r["problem_id"], {})
        out.append(f"**`{r['problem_id']}`** — {r['window_kind']} window at "
                   f"chunk {r['chunk_index']}, ‖activation‖ = "
                   f"{r.get('activation_norm', float('nan')):.1f}, "
                   f"gold answer `{t.get('gold')}`, trace's own answer "
                   f"`{t.get('final_answer')}`")
        out.append("")
        for s in r.get("samples", []):
            out.append(f"- *from the activation (sample {s['sample_index']}):* "
                       f"{_clip(s.get('explanation') or s.get('text'))}")
        for s in r.get("noise_samples", []):
            out.append(f"- *from Gaussian noise at matched norm:* "
                       f"{_clip(s.get('explanation') or s.get('text'))}")
        out.append("")
    return "\n".join(out)


def _summarise_test(name: str, r: dict) -> str:
    if r.get("status") != "ran":
        return str(r.get("reason", ""))[:140]
    if name == "F1_length":
        return (f"corr(length, tail fraction) = "
                f"{_fmt(r.get('corr_length_vs_tail_fraction'))}")
    if name == "F2_difficulty":
        return f"tail fraction by dataset: {r.get('by_dataset')}"
    if name == "F3_verbaliser_only":
        return (f"reproduced from noise {_fmt(r.get('reproduced_from_noise'))}; "
                f"cosine real {_fmt(r.get('mean_cosine_real'))} vs control "
                f"{_fmt(r.get('mean_cosine_control'))}")
    if name == "F4_parser":
        return (f"final-answer agreement {_fmt(r.get('final_answer_agreement'))}; "
                f"chunks {_fmt(r.get('mean_chunks_primary'), 1)} vs "
                f"{_fmt(r.get('mean_chunks_alternative'), 1)} under the "
                f"alternative segmentation")
    if name == "F5_ast_sensitivity":
        txt = (f"over {len(r.get('settings_exercised') or [])} settings, mean "
               f"tail fraction varies by "
               f"{_fmt(r.get('mean_tail_fraction_range'))}, non-trivial tail "
               f"rate by {_fmt(r.get('nontrivial_tail_rate_range'))}")
        na = r.get("settings_not_applicable") or {}
        if na:
            txt += f"; not testable: {', '.join(sorted(na))}"
        return txt
    if name == "F6_probe_leakage":
        parts = [f"real AUC {_fmt(r.get('real_auc'))}"]
        for scope, arm in (r.get("shuffles") or {}).items():
            tag = "" if arm.get("meaningful") else " (moved no labels - no control)"
            parts.append(f"{scope} {_fmt(arm.get('auc'))}{tag}")
        leak = r.get("leak_suspected")
        parts.append("leak suspected: "
                     + ("not testable" if leak is None else str(leak)))
        return "; ".join(parts)
    if name == "F7_layer_sweep":
        return "; ".join(f"L{k}: AUC {_fmt(v.get('auc'))}"
                         for k, v in (r.get("layers") or {}).items())
    if name == "F8_quantisation":
        return (f"mean cos(nf4, bf16) = {_fmt(r.get('mean_cosine'), 4)} over "
                f"{r.get('n_vectors')} vectors (5th pct "
                f"{_fmt(r.get('p05_cosine'), 4)})")
    if name == "F10_position":
        return "; ".join(f"{k}: {_fmt(v.get('mean_cosine'))}"
                         for k, v in (r.get("by_window_kind") or {}).items())
    return ""


if __name__ == "__main__":
    sys.exit(main())
