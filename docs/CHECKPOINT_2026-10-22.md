# rep1 checkpoint, 2026-10-22 (D92 pause, D93 resume)

The state of the confirmatory replication (D75) when it was resumed, the
parameters it runs under, and what must pass before its verdict is read.
Verified against the files, logs and manifest, not against earlier status
messages. **No registered result exists yet.**

## 1. Where the run is

| stage | state at 20:19 IST, 2026-10-22 | evidence |
|---|---|---|
| traces | complete: 400 rows, 400 ok, 400 unique, frozen order; 17 at the 768-token cap | `runs/rep1/manifest.json`, runbook trace gate |
| ast | complete: 380/400 with a tail (plan minimum 200) | manifest `ast.metrics.n_with_tail` |
| acts | complete: 1,200 blocks (layers 14, 20, 24), 0 failed, 0 clamped | manifest, acts gate |
| baselines | complete: rules `convergence`, `probe`; no semantic entropy, by design | manifest, baselines gate |
| NLA verbalisation | complete: 2,227/2,227 windows, unique; 4,439/4,454 samples pass integrity | `runs/rep1/nla/verbalisations.jsonl` |
| NLA reconstruction | **running from 0/2,227** (stopped at 2,000 at the D92 pause; the phase is not resumable) | `runs/rep1_runbook/nla.log` |
| menu forcing (48 tokens) | not started | no `runs/rep1_forced/provenance.json` |
| D75 confirmatory | not started | no `runs/rep1_o3_registered/` |
| D79 position/truncation audit | not started | no `runs/rep1_position_audit/` |
| post-rep1 queue | not started | no `runs/post_rep1_queue/queue_resume.out` |

## 2. Resume procedure (as run at 20:19:11)

1. Make sure nothing is running: no `python.exe` whose command line names
   `resume_rep1`, `rep1_runbook`, `post_rep1_queue` or a stage script.
2. Free the GPU (D89): stop `OmenCommandCenterBackground.exe` and its
   helpers (`HP.OMEN.VadRedirectHelper`, `HP.OMEN.VirtualCamHelper`,
   `omenmqtt`), or quit OMEN Command Center from the tray. Leave the HP
   services alone. `nvidia-smi` should read about 0 MiB.
3. Check that the machine is on AC power and that sleep is set to never.
4. From the pipeline's checkout (`D:\NLAutoencoders-AnswerStableTail`):

   ```powershell
   $env:PYTHONUTF8 = "1"
   .venv\Scripts\python.exe scripts\resume_rep1.py
   ```

   `resume_rep1.py` sets `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8`
   itself (D90). It checks the plan hash, skips complete stages and resumes
   partial ones.
5. Watch: `Get-Content runs\rep1_runbook\runbook_resume.out -Tail 10 -Wait`
   for stage transitions, `runs\rep1_runbook\nla.log` for NLA progress.
6. To pause: `scripts\safe_stop.ps1`. It stops the orchestrators. It does not
   stop a running NLA or forcing stage; it names it so you can decide.

## 3. Stages still to run, and the gate each must pass

The order and the gates are `scripts/rep1_runbook.py` and
`scripts/post_rep1_queue.py`. Each stops at the first failure.

| # | stage | gate (the code's check) |
|---|---|---|
| A | reconstruction | manifest `nla` complete with both phases; real-sample integrity ≥ 0.5 |
| B | `force_all_boundaries.py --config replication --menu-only --n-repro 200` | `block_size` 2; all 200 re-forced rows written. The plan's ≥ 0.95 agreement block is applied by the registered analysis |
| C | `reanalyse_o3_registered.py --config replication --forced-dir runs\rep1_forced --confirmatory docs\REPLICATION_PLAN.md --plan-sha256 e45629a7914fb4c8eefc95758014a72ab7af53c363e670c862f441c886a60761` | `o3_registered.json` exists and has no `blocked` key |
| D | `position_truncation_audit.py --config replication --forced-dir runs\rep1_forced --confirmatory` | `position_truncation.json` exists |
| E1 | rep1 secondaries: information audit (pre-specified and post-hoc), lexicon transfer, selective risk | each output JSON exists |
| E2 | D82 256-token forcing on rep1, its analysis (`--confirmatory`, labelled sensitivity) and the O256 information audit | forcing finished with full row count; analysis not blocked |
| E3 | Stage 1 (D83, approved as revised in D85): positions → verbalise → reconstruct → force → analyse | `rep1_stage1/analysis.json` |
| E4 | D82 on `main_nla240` (exploratory, data already seen; last by D87) | as E2 |

## 4. Parameters, and where each comes from

| parameter | value | source |
|---|---|---|
| target model | `Qwen/Qwen2.5-7B-Instruct`, NF4 | `configs/base.yaml` (`replication.yaml` is merged over it) |
| target layer / probe layer for H1 | 20 | `base.yaml`; plan §2 |
| NLA verbaliser / reconstructor | `kitft/nla-qwen2.5-7b-L20-av` / `-ar`, NF4 | `base.yaml` |
| verbaliser windows, samples, tokens | 2 per kind, 2 samples, 256 tokens; no Gaussian control | `replication.yaml`; plan §3 |
| problems | 400: 240 GSM8K, 160 MATH | `replication.yaml`; plan §3 |
| trace cap | 768 tokens, greedy | `replication.yaml` |
| AST | K = 5, T = 0.8, 200-token continuations | `replication.yaml`, `base.yaml`; plan §3 |
| registered outcome | forced answer, 48 tokens (O48) | `replication.yaml` `force_answer_max_new_tokens`; plan §3 |
| batch size | 2 (D78, deviation 1) | `replication.yaml` |
| instrument check | 200 re-forced boundaries, block at agreement < 0.95 (D80, deviation 2) | plan deviations; `rep1_runbook.py` `N_REPRO` |
| hypotheses | H1 readout − layer-20 probe; H2 readout − no signal | plan §2 |
| budgets | {0.1, …, 0.9} × B\*, tolerance 0.02, ≥ 3 matched | plan §4 |
| uncertainty | paired problem bootstrap, 2,000 replicates, seed 20261019; Holm over {H1, H2} at 0.05 | plan §4; `reanalyse_o3_registered.py` `REPS`, `SEED` |
| decision rule | supported / reversed / equivalent within ±0.05 / advantage ≥ 0.05 excluded / inconclusive | plan §4 |
| memory setting | `PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8` (engineering, D90) | `resume_rep1.py` |

`replication.yaml` also sets `analysis.bootstrap_iterations: 10000`; the
registered analysis does not read it (D93).

## 5. Hashes, recomputed 2026-10-22 in the pipeline's checkout

| file | recorded | where recorded | match |
|---|---|---|---|
| `docs/REPLICATION_PLAN.md` (as amended) | `e45629a7…0761` | D80, `rep1_runbook.py` | yes |
| `configs/replication.yaml` | `d31383f6…5f23` | D78 | yes |
| `runs/rep1/data/problems.jsonl` | `e633885f…d248` | D75 | yes |
| `scripts/rep1_runbook.py` | `5da4593c…11a7` | D85 | yes |
| `scripts/extended_forcing_analysis.py` | `849d8674…869a` | D85 | yes |
| `scripts/information_audit.py` | `7b26afe8…8d56` | D85 | yes |
| `docs/EXTENDED_FORCING_PLAN.md` (as amended) | `027f1bc0…d555` | D85 | yes |
| `docs/LEAKAGE_RESISTANT_DESIGN.md` | `514818cf…6afe` | D88 (replacing D85's `2bb536f2…7756`) | yes |
| `docs/POSITION_TRUNCATION_PLAN.md` (as amended) | `bffc9e08…c877` | D79 | yes |
| `docs/O3_REGISTERED_REANALYSIS_PLAN.md` | `02735946…9876` | D91 | yes; D69's pre-deviation hash `6e0cc75d…f429` cannot be reconstructed (D91) |
| `scripts/stage1_power_audit.py` | `64eaa58e…5716` | D85 | yes in the pipeline's checkout; `main` drops one unused import (D91) |
| `runs/stage1_power_audit/power.json` | `85300f39…18a1` | D85 | yes |

## 6. What the verdict will and will not mean

- The verdict is the plan's mechanical reading of H1 and H2 on O48. Nothing
  else enters it.
- D79's position-only control is reported beside it, not inside it. It says
  whether a positive H2 could be menu position rather than content.
- D82 (O256) and Stage 1 are sensitivity and exploratory work. They never
  replace the registered outcome.
- Stopping value says nothing about whether the explanations are faithful.
  The project makes no faithfulness claim.
