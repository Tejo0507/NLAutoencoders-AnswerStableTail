# 1. Introduction (draft v2, 2026-10-21)

*Status markers:*
- **[R]** registered;
- **[C]** confirmatory (the rep1 replication; computed, D94);
- **[S]** pre-specified secondary or sensitivity (rep1; D94–D96);
- **[X]** exploratory;
- **[P]** post hoc;
- **[A]** analytic.

*Every number is from `paper/CLAIM_EVIDENCE_MATRIX.md`. Bracketed
"pending" items must be filled from their analysis outputs, not edited by
hand.*

Reasoning models often keep writing after the answer they will give is
already settled [7, 8]. Stopping them earlier needs a signal that the
answer is done. Candidates include agreement between intermediate answers
[5], hidden-state probes for the correctness of an intermediate answer [6],
the model's confidence, and calibrated or learned stopping rules [28, 29].
Natural-language autoencoders (NLAs) [10] add a different kind of signal: a
textual description of the activation at a point in the trace, trained so
that the activation can be reconstructed from it. If such a description says
that the model is concluding rather than still setting up, it might tell us
when to stop. It would also say *why*.

We ask whether an NLA readout gives useful information about when a
model's answer is settled, **beyond what cheaper signals already give**. We
study Qwen2.5-7B-Instruct with its released layer-20 NLA, on GSM8K and MATH,
under a pre-registered plan. We report five things.

1. **The registered stopping comparison was uninformative, and the
   readout's apparent advantage after correcting it is not the readout's.**
   - The registered test compared rules at budgets above the oracle bound
     [R, read post hoc].
   - Rescored on the registered forced-answer outcome, the readout's score
     beat a rule with no signal [P].
   - A rule that uses no readout at all reproduces that advantage: stop at
     the first candidate point if it is early [X; pre-specified for the
     replication].
   - When the readout's rule stops, the stop is unsafe *more* often than
     chance [P].
   - A pre-registered replication on 400 unseen problems supports both
     registered hypotheses on the 48-token outcome [C].
   - A position-only rule beats the readout there [S].
   - Under a 256-token forcing budget, neither hypothesis is supported:
     both read *equivalent within ±0.05*, which excludes effects of 0.05
     or more but not smaller ones [S].
2. **The readout's score carries no stopping information beyond position
   and a probe** (ΔAUC −0.001 [−0.002, −0.000]) [X].
3. **The readout's *text* does carry information about the reasoning
   state.**
   - It encodes a phase, concluding against setting up, beyond the raw text
     of the trace (+0.073 AUC) [X].
   - But for predicting stopping safety it adds **no practically relevant
     information beyond a linear probe on the same activation, trained for
     the same label** (−0.001 [−0.004, 0.002]) [P].
   - At positions chosen without the outcome, the text's increment over
     that same-label probe, for predicting these stopping labels, is
     +0.00002 [−0.004, 0.004]: equivalent within ±0.01 [S, Stage 1]. The phase vocabulary transfers to unseen problems
     (+0.091 AUC) [S].
4. **Three evaluation choices can manufacture or erase an early-stopping
   advantage.** We demonstrate each with a control:
   - *Candidate points derived from the outcome* leak it.
   - *A matched-mean-saving estimand that counts abstention as safe* ranks
     an information-free rule above a probe with boundary AUC 0.86, even
     with no outcome-derived points [X].
   - *A short forced-answer budget* gives answers that are reproducible but
     invalid before the answer settles. With 48 tokens, 54–57% of pre-tail
     forced answers are cut off, and 98–99% of those disagree with the
     model's own continuations [X].

   - Given 256 tokens, 520 of the 600 answers cut off at 48 tokens *before
     the tail* (87%) reach the final answer.
   - Under that outcome the registered hypotheses are not supported [S].
     The registered 48-token verdict stands, but it depends on the forcing
     budget.

   Two recent early-exit methods force answers with 48- and 64-token
   budgets [28, 31].
5. **Cost.** One NLA check costs 0.83 of a mean trace in generated tokens,
   so verbalising at each candidate point cannot save compute in this
   setting [A].

**What we do not claim:**
- that the explanations are faithful to the computation (they are not
  evaluated semantically; §9);
- that cost-aware or risk-controlled evaluation is new (it is not
  [28, 29]);
- that the results hold beyond this model and its NLA.

The contribution is a careful negative result about NLAs as stopping
signals, explained by the evaluation choices that would otherwise have
suggested a positive one, and with those choices measured.
