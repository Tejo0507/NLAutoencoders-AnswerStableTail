# Run artifacts: NLA Answer-Stable Tail, with the confirmatory replication (DRAFT release notes)

> **Draft, not published.** For the GitHub release `data-runs-<date>`
> (`docs/ARCHIVE_PLAN_REP1.md` §8). Fill `<…>` from the final build; do not
> reuse candidate values.

This archive holds the stored outputs of the study and its pre-registered
confirmatory replication on 400 fresh problems (`rep1`), built from commit
`<COMMIT>` of https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail.

- **Archive:** `nlaast-runs-<date>.tar.gz`, SHA-256 `<SHA256>`
  (`<BYTES>` bytes); `<N>` run files, `<K>` with local path roots replaced
  (listed with their original hashes in `TRANSFORMS.json`).
- **What is new since `data-runs-2026-10-19`:** the replication and its
  pre-specified analyses (`rep1*`), the post-hoc and exploratory analyses
  of the main run (D68–D84, D96), the D76 rewording null
  (`main_paraphrase_llm*`, exploratory), and the hash manifest of the
  replication's code state.
- **Evidence classes:** only `runs/main/` (registered) and the rep1
  confirmatory directories carry registered or confirmatory results;
  `RUNS.json` classifies every directory. Nothing here establishes that the
  NLA explanations are faithful, and the selective-risk results are not
  deployment evidence.
- **Withheld:** the blinded human-annotation sheets and their unblinding key
  (`main_claim_annotation`), until the annotation is complete.
- **Verify:** `sha256sum -c SHA256SUMS.txt` inside the extracted directory,
  and the archive against the `.sha256` file published beside it.
- **Licence:** CC BY 4.0 for the artifacts; GSM8K and MATH problem text
  (MIT) and model outputs (Apache-2.0) as stated in `NOTICE.md`.
