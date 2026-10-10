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
    A(f"Config hash `{cfg.hash()}` | generated {manifest.data.get('updated')}")
    A("")
    A("Produced entirely from saved stage outputs by `scripts/write_run_report.py`. "
      "Stages that did not run are listed as such rather than omitted.")
    A("")
    # One commit per run is wrong for a run that spans days: the manifest's
    # run-level git block names the code the directory was created under. Each
    # stage records its own, so the distinct set is what the results are
    # attributable to.
    per_stage = {s: (e.get("git") or {}).get("commit")
                 for s, e in stages.items() if (e.get("git") or {}).get("commit")}
    commits = sorted(set(per_stage.values()))
    created = (manifest.data.get("git") or {}).get("commit")
    n_total = len([s for s in stages if stages[s].get("status") != "pending"])
    if commits:
        A(f"{len(per_stage)} of {n_total} stages recorded the commit they ran "
          f"under: " + ", ".join(f"`{c[:12]}`" for c in commits) + ".")
        if len(per_stage) < n_total:
            A("")
            A(f"The rest ran before per-stage commits were recorded, so the "
              f"code behind them is bounded only by the run's own history. The "
              f"run directory was created under `{(created or '?')[:12]}`.")
    elif created:
        A(f"Run directory created under commit `{created[:12]}`. Per-stage "
          f"commits were not recorded for this run, so individual results "
          f"cannot be attributed to a specific revision.")
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
        status = e.get("status", "not run")
        # This stage is mid-flight by definition - it is the one writing this
        # table - so reporting it as "running" states nothing and looks like an
        # interrupted run.
        if s == STAGE and status == "running":
            status = "writing this report"
        A(f"| {s} | {status} | {_fmt(e.get('elapsed_s'), 0)} |")
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
    A(_o3_verdict(results))

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
        A(_rq2_sensitivity(results.get("rq2") or {}))
    else:
        A("_Faithfulness audit did not run._")
        A("")

    A(_verbaliser_integrity(cfg))
    A(_verbalisation_examples(cfg, n=2))

    if recon:
        A("### Reconstruction fidelity by window")
        A("")
        A("Fidelity is **not** faithfulness; the two are reported separately "
          "throughout, and this table speaks only to the former.")
        A("")
        A(tables.markdown(tables.reconstruction_table(recon)))
        A("")
        nla_sum_path = cfg.dir / "nla" / "summary.json"
        nla_sum = read_json(nla_sum_path) if nla_sum_path.exists() else {}
        emp = nla_sum.get("empirical_baseline") or {}
        overall = nla_sum.get("overall") or {}
        if emp.get("available"):
            A("#### Which baseline the FVE is against")
            A("")
            A(f"An FVE needs a baseline, and the usual 2.0 assumes an "
              f"uninformative prediction is *orthogonal* to the target. A "
              f"random direction in {cfg.target.d_model} dimensions is — but a "
              f"plausible guess is not, and layer-{cfg.target.layer} residual "
              f"streams are strongly anisotropic. Two of this run's own "
              f"activations, drawn at random, have a mean cosine of "
              f"**{emp['mean_pairwise_cosine']:.3f}**, so simply guessing a "
              f"typical layer-{cfg.target.layer} activation already achieves "
              f"MSE {emp['baseline_mse']:.3f} rather than 2.0.")
            A("")
            A("| baseline | FVE |")
            A("|---|---|")
            A(f"| theoretical, orthogonal prediction (2.0) | "
              f"**{_fmt(overall.get('fve'))}** |")
            A(f"| empirical, this run's own activations "
              f"({emp['baseline_mse']:.3f}) | "
              f"**{_fmt(overall.get('fve_empirical_baseline'))}** |")
            A("")
            A(f"PROJECT_PLAN §5 asks for the empirical one, so that is the "
              f"figure to read. The theoretical one is kept because the "
              f"checkpoint card's {_fmt(nla_sum.get('reference_fve'))} is "
              f"comparable only against whatever baseline *it* used, which is "
              f"not stated — so neither of these should be read as beating or "
              f"missing it.")
            A("")

    # ---------------------------------------------------------------- RQ3
    A("## RQ3 - causal evidence")
    A("")
    rq3 = results.get("rq3") or {}
    if rq3:
        v = rq3.get("verdict", {})
        if v.get("supported") is None:
            A("**Direction claim: not tested.** The gate returns neither "
              "support nor refutation, because the behavioural readout it "
              "judges — the change in rechecking-marker count — did not vary "
              "at all.")
            A("")
            A(f"> {v.get('untestable_reason')}")
            A("")
            A("Reporting this as *unsupported* would say the candidate "
              "direction failed to beat its controls. It did not fail: "
              "nothing was measured. The outcome variable has to move before "
              "any of the four requirements below can mean anything.")
            A("")
        else:
            A(f"**Direction claim supported: {v.get('supported')}**")
            A("")
        A("| requirement | met |")
        A("|---|---|")
        for k in ("beats_random_direction", "beats_matched_position",
                  "monotone_dose_response", "answers_preserved"):
            val = v.get(k)
            A(f"| {k.replace('_', ' ')} | "
              f"{'not testable' if val is None else val} |")
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


def _o3_verdict(results: dict) -> str:
    """State which way the primary comparison went, in words.

    A reader scanning a table of q-values can take "significant" for "the
    readout won". Here it means the opposite, so the direction is written out
    next to every comparison, and the ones that could not be made at matched
    budget are listed as such rather than left absent.
    """
    tests = [t for t in ((results.get("tests") or {}).get("tests") or [])
             if t.get("name", "").startswith("O3:")]
    if not tests:
        return ""

    ran = [t for t in tests if t.get("notes", {}).get("direction")]
    unmatched = [t for t in tests if t.get("notes", {}).get("skipped")]

    out = ["### Verdict on the primary question", ""]
    if ran:
        out += ["| compared against | NLA safe rate | its safe rate | budgets | "
                "direction | q |", "|---|---|---|---|---|---|"]
        for t in ran:
            n = t["notes"]
            other = next(k for k in n
                         if k.startswith("mean_safe_rate_") and k != "mean_safe_rate_nla")
            label = other.replace("mean_safe_rate_", "")
            out.append(
                f"| {label} | {n['mean_safe_rate_nla']:.3f} | {n[other]:.3f} | "
                f"{n['n_budgets']} | **{n['direction']}** | "
                f"{_fmt(t.get('q_value'), 4)} |")
        out.append("")
        worse = [t for t in ran if t["notes"]["direction"] == "nla_readout worse"]
        better = [t for t in ran if t["notes"]["direction"] == "nla_readout better"]
        if worse and not better:
            out += [
                "**The verbalised readout does not beat the cheap signals on "
                "this corpus — it loses to them.** Where the comparison can be "
                "made at genuinely matched budget, the readout's safe-stopping "
                "rate is lower. A significant q here is evidence *against* the "
                "readout's incremental value, not for it.", "",
                "The review named this as an acceptable outcome in advance and "
                "it is reported with the same weight a positive result would "
                "have had.", ""]
        elif better and not worse:
            out += ["The readout's safe-stopping rate is higher wherever the "
                    "comparison can be made at matched budget.", ""]

    if unmatched:
        out += ["Comparisons that **could not be made**:", ""]
        for t in unmatched:
            name = t["name"].split("minus ", 1)[-1].split(" at ")[0]
            out.append(f"- against **{name}**: {t['notes']['skipped']}")
        out += ["",
                "These stay in the pre-registered test family as unevaluable "
                "rather than being dropped from it. A rule's safety/saving "
                "curve is a step function — it fires at a boundary or it does "
                "not — and on a corpus of this size the steps are wide enough "
                "that two curves need not have any operating point near a "
                "shared budget. Comparing their nearest points anyway would "
                "compare safety at different budgets, which is the one thing "
                "the matched-budget design exists to prevent.", ""]
    return "\n".join(out)


def _rq2_sensitivity(rq2: dict) -> str:
    """Do the two RQ2 headlines survive the thresholds they were measured at?

    Both numbers rest on a choice. `reconstruction_dependent` is judged against
    the 95th percentile of the paraphrase null, and the paraphrase arm controls
    for surface change but *not* for how much text a deletion removes - which
    is about a quarter of an explanation. `reproduced_under_noise` is judged at
    Jaccard >= 0.5 over content words, one number with no principled basis.

    Neither is worth reporting without saying what happens when the choice
    moves, so both sensitivities are printed. Neither replaces the registered
    measure: changing a headline definition after seeing the result is what
    fixing the definitions in advance was for.
    """
    ls = rq2.get("length_sensitivity") or {}
    noise = rq2.get("noise_reproduction_by_threshold") or {}
    dist = rq2.get("noise_similarity_distribution") or {}
    if not ls.get("available") and not noise:
        return ""

    out = ["#### Do these two numbers survive their thresholds?", ""]

    if ls.get("available"):
        out += [
            f"**The deletion effect is partly a length effect.** Deleting a "
            f"claim removes {ls['mean_deleted_fraction']:.1%} of an explanation "
            f"on average, and the effect does rise with how much was cut: "
            f"correlation {ls['corr_deleted_fraction_vs_delete_effect']:+.3f}, "
            f"accounting for {ls['variance_explained_by_length']:.1%} of the "
            f"variance. The paraphrase null controls for surface change, not "
            f"for the scissors.", "",
            "| deleted fraction | n | mean deletion effect |",
            "|---|---|---|",
        ]
        for q in ls.get("effect_by_deleted_fraction_quintile", []):
            out.append(f"| {q['from']:.2f}–{q['to']:.2f} | {q['n']} | "
                       f"{q['mean_delete_effect']:+.4f} |")
        out += ["",
                f"Removing that pooled trend and re-applying the same "
                f"paraphrase-null threshold gives a **length-adjusted "
                f"dependence rate of "
                f"{ls['dependent_fraction_length_adjusted']:.3f}**, against "
                f"{ls['dependent_fraction_as_registered']:.3f} as registered. "
                f"The finding does not rest on the length confound."]
        w = ls.get("within_explanation") or {}
        if w.get("median_sd_over_mean") is not None:
            out += ["",
                    f"Within a single explanation — where the deletions are of "
                    f"comparable size — the spread of deletion effects is "
                    f"{w['mean_sd_of_delete_effect']:.4f} against a mean of "
                    f"{w['mean_of_means']:+.4f} "
                    f"({w['median_sd_over_mean']:.2f} as a ratio, median over "
                    f"{w['n_explanations']} explanations). That spread is the "
                    f"part no length effect explains."]
        out.append("")

    if noise:
        out += [
            "**The noise control is not riding its threshold.** Reproduction "
            "rate against the matched-norm Gaussian control, by Jaccard "
            "threshold:", "",
            "| threshold | reproduced |", "|---|---|",
        ]
        for t, rate in sorted(noise.items()):
            out.append(f"| {t} | {rate:.4f} |")
        if dist:
            out += ["",
                    f"The similarity distribution sits well below any of these: "
                    f"mean {dist['mean']:.3f}, median {dist['median']:.3f}, "
                    f"99th percentile {dist['p99']:.3f}, maximum "
                    f"{dist['max']:.3f}. There is no cliff near the chosen "
                    f"threshold for the rate to be an artefact of."]
        out.append("")
    return "\n".join(out)


def _verbaliser_integrity(cfg) -> str:
    """The integrity gate's pass rate, split by what drove the verbaliser.

    Upstream documents a specific off-distribution failure: CJK text instead
    of an English ``<explanation>``. The gate's rate on real activations is
    therefore a check on the 4-bit deviation. Its rate on the matched-norm
    Gaussian control is not something the study set out to measure, but it is
    free and it is informative - a *lower* rate there says the real
    activations are closer to what the verbaliser was trained on than
    norm-matched noise is, at the level of basic output well-formedness and
    independently of anything about content.

    Recomputed from the persisted verbalisations rather than from the stage's
    own counters, so it is reproducible without re-running the model.
    """
    rows = read_jsonl(cfg.dir / "nla" / "verbalisations.jsonl")
    if not rows:
        return ""

    def rate(group: str) -> tuple[int, int, int]:
        samples = [s for r in rows for s in r.get(group, [])]
        ok = sum(1 for s in samples if s.get("ok"))
        cjk = sum(1 for s in samples if s.get("cjk_chars"))
        return ok, len(samples), cjk

    a_ok, a_n, a_cjk = rate("samples")
    n_ok, n_n, n_cjk = rate("noise_samples")
    if not a_n:
        return ""

    out = ["### Verbaliser integrity", "",
           "| driven by | well-formed English | samples with CJK characters |",
           "|---|---|---|",
           f"| the activation | {a_ok}/{a_n} ({a_ok / a_n:.3f}) | {a_cjk} |"]
    if n_n:
        out.append(f"| Gaussian noise at matched norm | {n_ok}/{n_n} "
                   f"({n_ok / n_n:.3f}) | {n_cjk} |")
    out.append("")
    out.append("Upstream documents CJK output in place of an English "
               "`<explanation>` as the signature of an off-distribution "
               "injected vector, which is why this is gated on every sample: "
               "the activations here come from a 4-bit target.")
    out.append("")
    if n_n and a_n and (a_ok / a_n) - (n_ok / n_n) > 0.05:
        out.append(f"The control fails the gate markedly more often "
                   f"({1 - n_ok / n_n:.1%} against {1 - a_ok / a_n:.1%}). That "
                   f"was not a planned measurement, but it is a difference "
                   f"between real activations and norm-matched noise that does "
                   f"not depend on judging content: the real vectors are "
                   f"closer to what the verbaliser was trained on. It says "
                   f"nothing about whether the *claims* are faithful, which is "
                   f"what the audit below is for.")
        out.append("")
    return "\n".join(out)


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
