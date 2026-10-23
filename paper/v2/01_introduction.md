# 1. Introduction (draft v3, 2026-10-23)

*Status markers:*
- **[R]** registered;
- **[C]** confirmatory (the rep1 replication; computed, D94);
- **[S]** pre-specified secondary or sensitivity (rep1; D94–D96);
- **[X]** exploratory;
- **[P]** post hoc;
- **[A]** analytic.

*Every number is from `paper/CLAIM_EVIDENCE_MATRIX.md`, and every claim is
stated no more strongly than its row allows.*

Reasoning models often keep writing after the answer they will give is
already settled [7, 8]. Stopping them earlier needs a signal that the
answer is done. Candidates include agreement between intermediate answers
[5, 34], the confidence of a trial answer [33], next-token entropy after an
end-of-thinking token [36], hidden-state probes and neurons [6, 37, 38],
and calibrated or learned stopping rules [28, 29]. Natural-language
autoencoders (NLAs) [10] add a different kind of signal: a textual
description of the activation at a point in the trace, trained so that the
activation can be reconstructed from it. If such a description says that
the model is concluding rather than still setting up, it might tell us when
to stop. It would also say *why*.

We ask whether an NLA readout gives useful information about when a
model's answer is settled, **beyond what cheaper signals already give**. We
study Qwen2.5-7B-Instruct with its released layer-20 NLA, on GSM8K and MATH,
under a pre-registered plan and a confirmatory replication on 400 unseen
problems. We report five things.

1. **The registered replication supports both hypotheses on the registered
   48-token outcome, and two pre-specified analyses qualify it.**
   - On the 48-token forced-answer outcome, the readout beats the
     registered correctness probe (H1, +0.047) and a rule with no signal
     (H2, +0.057) [C].
   - A position-only rule beats the readout by 0.025, and with its
     position trend removed the readout no longer beats no signal. So the
     advantage cannot be attributed to the content of the explanations [S].
   - Under a 256-token forcing budget, neither hypothesis is supported.
     Both read *equivalent within ±0.05*, which excludes effects of 0.05 or
     more but not smaller ones [S]. The registered 48-token verdict stands
     as registered; it is a result about commitment within 48 tokens.
   - On the earlier main run, the registered comparison had been
     uninformative [R, read post hoc], and a rule with no readout matched
     the readout's corrected advantage [X].
2. **The readout's score adds no practically relevant stopping information
   beyond position and the registered probe:** −0.001 [−0.002, −0.000] on
   the main run [X] and +0.0005 [−0.0015, 0.0025] on the replication [S].
3. **The readout's *text* does carry predictive information about the
   reasoning state, but no practically relevant increment over a cheaper
   read of the same activation.**
   - It distinguishes a phase, concluding against setting up, beyond the
     raw text of the trace (+0.073 AUC [X]; +0.077 on the replication
     [S]). A phase vocabulary frozen on earlier data transfers to unseen
     problems (+0.091 AUC) [S].
   - At positions chosen without the outcome, its increment over position
     plus a linear probe on the same activation, trained for the same
     label, is +0.00002 [−0.004, 0.004] AUC: *equivalent within ±0.01*
     [S, Stage 1].
   - These are statements about prediction and incremental value. We do
     not test whether the explanations are faithful to the computation.
4. **Three evaluation choices can manufacture or erase an early-stopping
   advantage.** We demonstrate each with a control:
   - *Candidate points derived from the outcome* leak it [A; §4.2].
   - *A matched-mean-saving estimand that counts abstention as safe* ranks
     an information-free rule above a probe with boundary AUC 0.86, even
     with no outcome-derived points [X].
   - *A short forced-answer budget* gives answers that are reproducible but
     not valid before the answer settles. With 48 tokens, 54–57% of
     pre-tail forced answers are cut off, and 98–99% of those disagree with
     the model's own continuations [X]. Given 256 tokens, 520 of the 600
     pre-tail cut-offs on the replication (86.7%) reach the final answer
     [S]. Two recent early-exit studies force answers with 48- and 64-token
     budgets [28, 31].
5. **Cost.** One NLA check costs 0.83 of a mean trace in generated tokens,
   so verbalising at each candidate point cannot save compute in this
   setting [A].

**What we do not claim:**
- that the explanations are faithful to the computation (they are not
  evaluated semantically or causally; §6);
- that any stopping rule here is safe to deploy: the selective-risk results
  are on an outcome-selected candidate menu (§4.5), and the stopping-policy
  study that would test deployment is proposed, not run (§5.4);
- that cost-aware or risk-controlled evaluation is new (it is not
  [28, 29, 38]);
- that the results hold beyond this model and its NLA.

## 1.1 Contribution statement

Prior work already establishes that stopping signals should be charged for
their compute and compared at matched risk [28, 29]; that hidden-state
probes predict the correctness of intermediate answers [6]; that
verbalizer evaluations need input and same-activation probe controls [12];
that truncate-and-answer reveals when answers commit [16, 31]; and that
answer agreement shows persistence, not finished reasoning [22]. We claim
none of these. To our knowledge, and within the focused review of §2, this
paper contributes:

1. **A pre-registered, confirmatory test of an NLA readout as a stopping
   signal, reported with its qualifications.** The registered hypotheses
   are supported on the registered 48-token outcome [C]. Pre-specified
   analyses show that the advantage cannot be attributed to explanation
   content and that the hypotheses are not supported under a 256-token
   outcome [S].
2. **A measurement of the forced-answer instrument's budget dependence.**
   At 48 tokens most pre-tail forced answers are cut off; at 256 tokens
   86.7% of them reach the final answer (88.1% on a second, non-independent
   arm), and the registered conclusion changes with the budget [S, X].
3. **Two demonstrated ways an evaluation can manufacture a stopping
   advantage:** candidate points derived from the outcome, and a
   matched-mean-saving estimand that counts abstention as safe [A, X].
4. **An incremental-value test of NLA text for a reasoning-state label
   against a same-activation, same-label probe**, using Li et al.'s
   control [12] at positions chosen without the outcome. The text predicts
   the labels, but adds no practically relevant information beyond the
   probe, within ±0.01 AUC [S].

All claims are bounded to one target model, one layer and one released NLA,
and to predictive content. There is no faithfulness claim.

The contribution is a careful negative result about NLAs as stopping
signals, together with the evaluation choices that would otherwise have
suggested a positive one, each of them measured.
