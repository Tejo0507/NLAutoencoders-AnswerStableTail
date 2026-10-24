"""Tables 1-3 of the revised manuscript, from the configs, the decision record
and stored analysis outputs only. CPU.

Every number is copied from a JSON written by an analysis script; nothing is
recomputed or rounded differently from how the record states it (3 decimals,
U+2212 for minus). Tables are numbered in order of first citation.

- Table 1: the setting and the pre-registration timeline (configs, DECISIONS.md).
- Table 2: the D75 confirmatory verdict verbatim
  (`runs/rep1_o3_registered/o3_registered.json`), its readings under the
  256-token outcomes (`runs/rep1_extforced/analysis.json`, D82), the D79
  position readings (`runs/rep1_position_audit/position_truncation.json`) and
  Stage 1's primary (`runs/rep1_stage1/analysis.json`).
- Table 3: the frozen phase lexicon (`runs/main_nla240_information/lexicon.json`,
  D84) and its out-of-sample transfer to rep1
  (`runs/rep1_information/information_lexicon.json`, D95).

    python scripts/make_paper_tables_v2.py --runs D:/elsewhere/runs

`--runs` is read only. The output is `paper/v2/tables_generated.md` in this
checkout; a table whose input is missing is reported and left out.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "v2" / "tables_generated.md"
MINUS = "\u2212"


def num(x: float, nd: int = 3, signed: bool = True) -> str:
    s = f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"
    return s.replace("-", MINUS)


def ci(c) -> str:
    return "—" if c is None else "[" + ", ".join(num(v, signed=False) for v in c) + "]"


def _load(runs: Path, rel: str):
    p = runs / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


#: The preregistration timeline: each decision's role, in the paper's words.
#: The date comes from the decision's own "Recorded" line in DECISIONS.md.
TIMELINE = (
    ("D66", "registered report frozen", "main run complete"),
    ("D68", "post-hoc audit of the frozen report", "after main's results"),
    ("D69", "O3 rescored on the registered outcome; plan fixed first (D91: the amended "
     "plan's hash was not recorded before the data)", "post hoc on main"),
    ("D75", "**confirmatory replication frozen** (plan, problems, config)", "before any rep1 trace"),
    ("D79", "position-only control and truncation audit specified", "before any rep1 outcome"),
    ("D78", "deviation 1: batch size 2", "before any rep1 outcome"),
    ("D80", "deviation 2: 200 re-forced boundaries; forcing in blocks of 2", "before any rep1 outcome"),
    ("D81", "information audit specified", "before any rep1 outcome"),
    ("D82", "256-token forcing pass specified", "before any extended answer"),
    ("D83", "leakage-resistant design (Stage 0, Stage 1)", "before any Stage 1 data"),
    ("D85", "Stage 1 approved as revised", "before any Stage 1 data"),
    ("D94", "**D75 verdict computed**; D79 audit", "outcomes"),
    ("D95", "secondaries, D82 on rep1", "outcomes"),
    ("D96", "Stage 1 and the final D82 pass", "outcomes"),
)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def table1(root: Path = ROOT) -> str | None:
    import re

    import yaml
    try:
        base = yaml.safe_load((root / "configs" / "base.yaml").read_text(encoding="utf-8"))
        cfg = {n: _deep_merge(base, yaml.safe_load((root / "configs" / f"{n}.yaml")
                                                  .read_text(encoding="utf-8")))
               for n in ("main", "replication")}
        dec = (root / "docs" / "DECISIONS.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    m, r = cfg["main"], cfg["replication"]
    rows = [
        ("target model", f"{m['target']['repo_id']}, {m['target']['precision'].upper()}",
         f"{r['target']['repo_id']}, {r['target']['precision'].upper()}"),
        ("problems", f"{m['data']['n_gsm8k']} GSM8K + {m['data']['n_math']} MATH",
         f"{r['data']['n_gsm8k']} GSM8K + {r['data']['n_math']} MATH, fresh (D75)"),
        ("MATH subjects", ", ".join(m["data"]["math_subjects"]),
         ", ".join(r["data"]["math_subjects"])),
        ("traces", f"greedy, ≤ {m['generation']['max_new_tokens']} tokens",
         f"greedy, ≤ {r['generation']['max_new_tokens']} tokens"),
        ("answer-stable tail", f"K = {m['ast']['k_continuations']} continuations at T = "
         f"{m['ast']['continuation_temperature']}, {m['ast']['continuation_max_new_tokens']} tokens",
         f"K = {r['ast']['k_continuations']} at T = {r['ast']['continuation_temperature']}, "
         f"{r['ast']['continuation_max_new_tokens']} tokens"),
        ("registered outcome", f"{m['generation']['force_answer_max_new_tokens']}-token forced answer (O48)",
         f"{r['generation']['force_answer_max_new_tokens']}-token forced answer (O48); 256-token "
         "sensitivity (O256, D82)"),
        ("probe and NLA layer", str(m["target"]["layer"]), str(r["target"]["layer"])),
        ("NLA verbaliser / reconstructor", f"{m['nla']['av_repo']} / -ar, "
         f"{m['nla']['precision'].upper()}", f"{r['nla']['av_repo']} / -ar, "
         f"{r['nla']['precision'].upper()}"),
        ("verbalisation", f"{m['nla']['n_samples']} samples × {m['nla']['max_new_tokens']} tokens, "
         f"{m['nla']['windows_per_problem']} per window kind",
         f"{r['nla']['n_samples']} samples × {r['nla']['max_new_tokens']} tokens, "
         f"{r['nla']['windows_per_problem']} per window kind"),
        # main.yaml's nla.max_problems is 80; the run used a --set override (D48)
        ("NLA arm", "120 problems (`--set nla.max_problems=120`, D48)",
         "every problem with a tail (380 of 400)"),
        ("batch size", str(m["generation"]["batch_size"]),
         f"{r['generation']['batch_size']} (D78, deviation 1)"),
    ]
    out = ["## Table 1. Setting and pre-registration timeline", "",
           "**(a) Setting** (from `configs/base.yaml` merged with `configs/main.yaml` and "
           "`configs/replication.yaml`; the NLA arm size of `main` is the D48 run override).", "",
           "| | `main` (registered study) | `rep1` (confirmatory replication) |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    out += ["", "**(b) Timeline** (dates from each decision's *Recorded* line in "
            "`docs/DECISIONS.md`).", "",
            "| decision | recorded | what it fixed | relative to the data |", "|---|---|---|---|"]
    for d, what, rel in TIMELINE:
        mm = re.search(rf"^## {d}\..*?\n\n\*\*Recorded ([^*]+?)[.,]?\*\*", dec, re.M | re.S)
        when = mm.group(1).split(",")[0].strip() if mm else "see record"
        out.append(f"| {d} | {when} | {what} | {rel} |")
    return "\n".join(out)


def table3(runs: Path) -> str | None:
    lex = _load(runs, "main_nla240_information/lexicon.json")
    tr = _load(runs, "rep1_information/information_lexicon.json")
    if lex is None or tr is None:
        return None
    n = max(len(lex["positive"]), len(lex["negative"]))
    rows = ["| # | concluding term (weight) | setting-up term (weight) |", "|---|---|---|"]
    for i in range(n):
        p = lex["positive"][i] if i < len(lex["positive"]) else None
        q = lex["negative"][i] if i < len(lex["negative"]) else None
        cell = lambda t: "" if t is None else f"{t[0]} ({num(t[1], 2, signed=False)})"  # noqa: E731
        rows.append(f"| {i + 1} | {cell(p)} | {cell(q)} |")
    out = [
        "## Table 3. The frozen phase lexicon and its transfer to unseen problems",
        "",
        f"Frozen {lex['frozen']} from the 240-problem arm (D84; source SHA-256 "
        f"`{lex['source_sha256'][:8]}…`). Exploratory in sample; the transfer to rep1 is "
        f"pre-specified (D81 deviation 3) and out of sample: "
        f"{tr['n_problems_shared_with_lexicon_source']} problems shared. "
        "It describes a reasoning *phase*; it is not a faithfulness claim.",
        "",
        *rows,
        "",
        "| rep1 label | units | lexicon AUC | position-only AUC | position + lexicon − position: ΔAUC [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for key, name in (("Y_tail", "tail membership"), ("Y_reg", "registered safety (O48)")):
        y = tr[key]
        d = y["pos+lex minus pos"]
        out.append(f"| {name} | {y['n_units']} | {num(y['lexicon_score']['auc'], signed=False)} "
                   f"| {num(y['prefix_tokens_only']['auc'], signed=False)} "
                   f"| {num(d['delta_auc'])} {ci(d['ci95'])} |")
    return "\n".join(out)


#: Readable names for the D79 comparisons; the stored keys are the code's.
POSITION_LABELS = {
    "readout - position_tokens": "readout − rising-position rule",
    "position_tokens - no_signal": "rising-position rule − no signal",
    "probe - position_tokens": "registered probe − rising-position rule",
    "readout_residualised - no_signal": "position-residualised readout − no signal",
    "readout - position_tokens_falling": "readout − falling-position rule",
    "position_tokens_falling - no_signal": "falling-position rule − no signal",
}


def table2(runs: Path) -> str | None:
    o3 = _load(runs, "rep1_o3_registered/o3_registered.json")
    ext = _load(runs, "rep1_extforced/analysis.json")
    pos = _load(runs, "rep1_position_audit/position_truncation.json")
    st1 = _load(runs, "rep1_stage1/analysis.json")
    if None in (o3, ext, pos, st1):
        return None
    c = o3["confirmatory"]
    out = [
        "## Table 2. The registered verdict and what qualifies it (rep1)",
        "",
        f"N = {o3['design_A_shared_menu']['n_problems']} problems with a tail; re-forcing "
        f"agreement {num(o3['instrument']['reproducibility_agreement'], signed=False)} "
        "(gate 0.95); paired problem bootstrap, 2,000 replicates; Holm over {H1, H2}; margin 0.05.",
        "",
        "**(a) Confirmatory, verbatim (D75, D94).** The registered outcome: 48-token forced answers.",
        "",
        "| hypothesis | Δ | 95% CI | p | Holm p | reading |",
        "|---|---|---|---|---|---|",
    ]
    names = {"H1_readout_minus_probe": "H1: readout − registered probe",
             "H2_readout_minus_no_signal": "H2: readout − no signal"}
    for k, v in c.items():
        out.append(f"| {names[k]} | {num(v['gap'])} | {ci(v['ci95'])} | {v['p']} "
                   f"| {v['p_holm']} | **{v['reading']}** |")
    out += ["", "**(b) Sensitivity, never replacing (a) (D82, D95).** f = "
            f"{num(ext['cut_off_meaning']['f_256_equals_final'], signed=False)}: "
            f"*{ext['cut_off_meaning']['reading']}*.", "",
            "| outcome | H1 Δ [95% CI] | H1 reading | H2 Δ [95% CI] | H2 reading |",
            "|---|---|---|---|---|"]
    #: Ocont, the continuations' majority, is a diagnostic outcome (D82).
    for cell in ("O48/registered", "O256/registered", "O256-u/registered", "Ocont/registered"):
        s = ext["cells"][cell]["d75_sensitivity"]
        h1, h2 = s["H1_readout_minus_probe"], s["H2_readout_minus_no_signal"]
        out.append(f"| {cell.split('/')[0]} | {num(h1['gap'])} {ci(h1['ci95'])} | {h1['reading']} "
                   f"| {num(h2['gap'])} {ci(h2['ci95'])} | {h2['reading']} |")
    g = pos["registered_outcome"]["gaps"]
    ro = pos["registered_outcome"]
    out += ["", "**(c) Pre-specified secondary: candidate position (D79, D94).** Not in the verdict.", "",
            "| comparison | Δ [95% CI] | Holm p |", "|---|---|---|"]
    for fam, keys in (("holm_X", ("readout - position_tokens", "position_tokens - no_signal",
                                  "probe - position_tokens", "readout_residualised - no_signal")),
                      ("holm_Y", ("readout - position_tokens_falling",
                                  "position_tokens_falling - no_signal"))):
        for k in keys:
            out.append(f"| {POSITION_LABELS[k]} | {num(g[k]['gap'])} {ci(g[k]['ci95'])} "
                       f"| {ro[fam][k]['p_holm']} |")
    out += ["", f"Readings: family X *{ro['position_reading']['overall']}*; family Y "
            f"*{ro['falling_reading']['overall']}*.", "",
            "**(d) Pre-specified, outcome-blind positions (Stage 1; D83, D85, D96).**", ""]
    p = st1["primary"]
    out.append(f"{st1['n_units']:,} positions in {st1['n_problems']} problems. Δ₁ = text over "
               "position + a same-label probe on the same layer-20 activation, 256-token label: "
               f"{num(p['delta'], 5)} {ci(p['ci95'])}, *{p['reading']}*. "
               "A statement about predictive information, not about faithfulness.")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    parts = ["# Tables 1-3 (generated)", "",
             "*Generated by `scripts/make_paper_tables_v2.py` from the configs, the decision "
             "record and stored outputs. Do not edit by hand; regenerate.*"]
    for name, f in (("table1", lambda _runs: table1()), ("table2", table2), ("table3", table3)):
        # in order of first citation: setting, verdict, lexicon
        t = f(a.runs)
        print(f"{name}: {'written' if t else 'skipped (input missing)'}")
        if t:
            parts += ["", t]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
