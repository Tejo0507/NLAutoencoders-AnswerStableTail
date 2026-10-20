# Focused novelty review (backlog E4), 2026-10-20

**Scope.** Two rounds of web search (standard, then extended) and abstract
pages of the closest neighbours. **Abstracts only.** Rows marked *full-text
check pending* must be read in full before the paper claims novelty there.
This is not a systematic review. It bounds what the paper may claim now.

## Closest prior work

| work | what it does | bearing on us |
|---|---|---|
| Dong, Qin, Shah, *When Does Learning to Stop Help? A Cost-Aware Study of Early Exits in Reasoning Models*, arXiv 2606.30852 (Jun–Jul 2026) | Compares confidence, entropy, answer-stability and learned stoppers at a fixed lost-correct risk (α = 0.15) over 18 settings. Shows that savings depend on serving cost: 32% saved under KV forking becomes 121% *extra* under repeated prefilling. Criticises post-hoc operating points | **Pre-empts** two general points: charge the signal's compute, and compare at matched risk, not post-hoc thresholds. We cite it, and present the 0.83-of-a-trace NLA check cost as an instance, not a discovery. *Full-text check pending:* whether it discusses outcome-derived candidate points or forced-answer validity |
| Wu, Zhou, Bates, Jaakkola, *Thought calibration*, EMNLP 2025 (arXiv 2505.18404) | Hidden-state probes predict whether the reasoning tree will still change. Stopping is risk-controlled with Learn-then-Test | Risk-controlled stopping already exists. Our selective-risk view is a diagnostic, not a new method |
| Zhang et al., *Reasoning Models Know When They're Right*, arXiv 2504.05419 | Correctness probes on hidden states used for early exit (24% of tokens saved); look-ahead information | Our registered probe follows this design. Our Stage 0 shows that a *same-label* probe is the stronger baseline |
| Li, Ceballos Arroyo, Rogers, Saphra, Wallace, *Do Activation Verbalization Methods Convey Privileged Information?*, ICML 2026 (arXiv 2509.13316) | Verbalizer benchmarks are solvable without the target's internals; verbalizations often reflect the verbalizer's own knowledge. Calls for controls | **The closest neighbour to direction D.** They control for the *input*. We add a control they do not report (abstract level): a **linear probe on the same activation, trained for the same label**. Our result agrees in spirit (no privileged information beyond a cheap read of the state) but differs in detail: NLA text *does* beat the raw input text (+0.073 AUC) |
| Lanham et al. 2023, *Measuring Faithfulness in Chain-of-Thought Reasoning*; *Reasoning Theater*, arXiv 2603.05488 (2026); Scalena et al., *Beyond the Commitment Boundary*, arXiv 2606.13603 (2026) | Truncate-and-answer ("early answering") and logit or probe reading of when the answer commits; early exit up to 55% shorter | They use the instrument we critique. None of the abstracts discusses the forced-answer **budget** or the validity of cut-off answers. *Full-text check pending* |
| LatentQA (arXiv 2412.08686), Patchscopes, SelfIE | Activation decoders; LatentQA reports +32% over linear probes on its tasks | The opposite direction to our C4. Their probes are generic, not same-label probes on our task. Must be discussed |
| Anthropic, *Natural Language Autoencoders* (transformer-circuits.pub/2026/nla) | The method we evaluate | — |

## Full-text checks done (2026-10-20, same day)

- **LearnStop (2606.30852)** uses the very instrument our A-series findings
  concern:
  - "greedily decode a short answer of at most A tokens", **A = 48 by
    default**;
  - nothing on what happens when that answer is cut off at the cap.

  Its checkpoints are a fixed token grid ([0, 128, …, 1536]), chosen without
  the outcome, so it has **no** menu-style leakage. Its frontier plots
  interpolate accuracy "at the same mean thinking-token cost". Our
  abstention/mean-saving finding bears on that view, though its headline
  comparison is risk-matched. It reports no position-only baseline.
- **Li et al. (2509.13316)** **do** train a logistic probe on the same
  activations for the same labels, but only in §5.3 (PersonaQA attributes),
  where it is "roughly on par" with LIT. Their main controls are input-text
  zero-shot and inversion. So the same-activation probe control **is not
  new**. What is new is its use on NLAs and on a reasoning-state label, with
  the result that the text beats the raw input text but not the probe.
- **Reasoning Theater (2603.05488)** forces answers with an 11-token budget
  on multiple-choice letters, where cut-off is unlikely. It notes forced
  answering can be "off-policy" for smaller models, but has no truncation
  analysis. Our validity critique concerns free-form answers that need
  derivation, where 48 tokens is often cut off.

## Added 2026-10-21: a reference already in `paper/references.md`

**Mo, Zhao, Geng, *Stable Answers, Unfinished Reasoning: Why Self-Consensus
Is Not a Safe Early-Exit Signal*, arXiv 2609.09989 (2026), [22].**
- **What it does:** a preregistered sweep of 3,520 consensus rules,
  concluding that agreement shows an answer "persists under a fixed probing
  procedure, not that reasoning has finished". At a 32% saving, one stop in
  nine lands on an answer the trajectory later abandons.
- **What it pre-empts:** our reading of the tail (A2) as recoverability,
  not redundancy. The paper must credit it there.
- **What it does not cover:** fixed forced-answer token budgets, hidden-state
  probes, or how candidate points are chosen.

## What the paper may claim, given this review

| claim | status |
|---|---|
| Charge the cost of the stopping signal; compare at matched risk | **Not ours.** Cite LearnStop and Thought Calibration |
| A matched-*mean*-saving estimand with abstention counted safe can rank an information-free rule above an AUC-0.86 probe (demonstrated) | **Plausibly new as a demonstration.** The remedy (risk control) is not new |
| Candidate points chosen from the outcome (design A's menu) leak it | **Plausibly new.** Full-text check of the two closest papers pending |
| Short forced-answer budgets give reliable but invalid answers before the tail; measured with a 256-token pass | **Plausibly new, and directly relevant**: LearnStop (2026) uses the same 48-token forced answer with no truncation analysis. D82 measures the effect |
| NLA text encodes reasoning phase, beats the raw input text, but adds nothing beyond a same-activation, same-label linear probe | **New for NLAs and for a reasoning-state label.** The control itself is Li et al.'s (§5.3) and must be credited to them. It qualifies LatentQA's reported advantage over generic probes |

## Remaining before submission

1. ~~Full text of 2606.30852 and 2509.13316~~ (done, above).
2. ~~Full text of 2606.13603~~ (done).
   - It **does** force generation at every sentence boundary, with the suffix
     "Therefore, the final answer is \boxed{", greedy decoding, and a
     continuation "capped at 64 tokens".
   - It calls the suffix "a soft intervention" and discards first-token
     collisions (about 9%, up to 57%), but reports no analysis of answers cut
     off at the cap.
   - Its candidate points are every boundary, so there is no menu-style
     leakage. Outcome information enters through probe labels and `i*`, not
     through point choice.
   - So **two 2026 early-exit papers use 48- and 64-token forced answers
     without a truncation analysis**. Our A-series and D82 results bear
     directly on both.
3. One more sweep for "overthinking" benchmarks (for example ThinkBrake,
   DEER, Dynasor) on how their stop points are chosen.
