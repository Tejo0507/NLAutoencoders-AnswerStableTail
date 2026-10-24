# Stopping-policy pilot: runbook (prepared 2026-10-24)

> **Not authorised to run.** This runbook prepares the pilot of
> `docs/STOPPING_POLICY_PILOT_PROTOCOL.md`. Nothing below may start until the
> researcher has made decisions D1–D9 (protocol §5), the protocol file is
> frozen and committed, its SHA-256 is recorded in `docs/DECISIONS.md`, and
> the GPU run is explicitly approved. `scripts/stopping_pilot.py` enforces the
> first two: it refuses to run on a draft or incomplete protocol.

## 0. What has been verified (2026-10-24, CPU only)

| item | state |
|---|---|
| Fresh pool | 909 GSM8K and 1,401 MATH test problems unused by any run (re-checked read-only) |
| Code | `scripts/stopping_pilot.py` and `src/nlaast/analysis/stopping_pilot.py`; 17 tests on synthetic runs (protocol gate, phase order, SGR calibration, evaluation, truncation rules) |
| `fit` on real data | dry run on stored rep1 Stage 1 (CPU, scratch output, test-only protocol): see §5 |
| GPU | RTX 4050 Laptop, 6,140 MiB; CUDA available; PyTorch 2.9.1+cu128, bitsandbytes 0.50.2, Transformers 5.18.0. **5,072 MiB free** with OMEN Command Center running, against about 5,552 MiB allocated by the NF4 target model (D76 log): the run would spill to shared memory (D89) unless OMEN is quit |
| Disk | D: 30.7 GB free |
| Weights | `D:\nla_models\nf4\Qwen__Qwen2.5-7B-Instruct` present (no download, no conversion) |

## 1. Compute estimate (measured rates; to be re-measured on the first problems)

| step | rate used | per problem |
|---|---|---|
| greedy trace, no tail scan (`ast.max_boundaries_evaluated: 0`) | about 329 tokens at 15–25 tokens/s single-stream | 13–22 s |
| 256-token forcing at every boundary | 11.1 boundaries per trace (rep1); about 8 s per block of 2 (Stage 1's forcing log, same budget and call) | about 45 s |
| layer-20 activations | about 1.5 s (rep1) | 1.5 s |
| re-forcing check | 50 boundaries per set | about 4 min per set |

About **60–70 GPU-seconds per problem without spill**, plus model loads:
**about 5–6 GPU-hours for N_C = 300, and 2.5–3 for N_E = 150; about 8–9 in
all.** With the VRAM spill of D89, expect several times longer. These are
estimates; the log reports the real rate from the first blocks.

## 2. Isolated tree (one-time, before the run)

All commands in PowerShell, from the new tree's `repo` directory unless
stated. `<COMMIT>` is the commit that records the frozen protocol.

```powershell
$E = "D:\NLAutoencoders-experiments\stopping-pilot-<date>"   # must not exist
New-Item -ItemType Directory $E | Out-Null
git clone https://github.com/Tejo0507/NLAutoencoders-AnswerStableTail.git "$E\repo"
git -C "$E\repo" checkout --detach <COMMIT>
$P = "D:\NLAutoencoders-AnswerStableTail"                      # read only
# raw datasets and every earlier run's problem list (for fresh-problem exclusion)
Copy-Item -Recurse "$P\data\raw" "$E\repo\data\raw"
Get-ChildItem "$P\runs\*\data\problems.jsonl" | ForEach-Object {
  $d = "$E\repo\runs\$($_.Directory.Parent.Name)\data"; New-Item -ItemType Directory -Force $d | Out-Null
  Copy-Item $_.FullName $d }
# rep1 training inputs for `fit` (traces, ast, acts, Stage 1)
foreach ($r in "rep1\traces", "rep1\ast", "rep1\acts", "rep1_stage1") {
  Copy-Item -Recurse "$P\runs\$r" "$E\train\$r" }
```

Then hash every copied file against its source (as for D76) and save the
lists in `$E\logs\`. Environment for every step:

```powershell
$env:NLAAST_QUANT_DIR = "D:\nla_models\nf4"; $env:HF_HUB_OFFLINE = "1"; $env:TRANSFORMERS_OFFLINE = "1"
$env:PYTORCH_ALLOC_CONF = "garbage_collection_threshold:0.8"; $env:PYTHONUTF8 = "1"
$PY = "$P\.venv\Scripts\python.exe"
```

## 3. Steps, in order (each resumable; each writes its own log)

| # | command (from `$E\repo`) | expected output | gate before the next step |
|---|---|---|---|
| 0 | quit OMEN Command Center (D89); `nvidia-smi` | no other process holding VRAM | ≥ 5.7 GB free |
| 1 | `& $PY scripts\stopping_pilot.py check` | protocol SHA-256 = the recorded one | match |
| 2 | `& $PY scripts\prepare_fresh_problems.py --config pilot_cal` | `runs\pilot_cal\data\problems.jsonl` (N_C ids) | 0 overlap reported |
| 3 | `& $PY scripts\generate_traces.py --config pilot_cal --no-entropy` | `runs\pilot_cal\traces\traces.jsonl` | ok traces ≥ 75% |
| 4 | `& $PY scripts\extract_activations.py --config pilot_cal --layers 20` | `runs\pilot_cal\acts\` | 0 clamped positions |
| 5 | `& $PY scripts\stopping_pilot.py force --run pilot_cal` | `runs\pilot_cal\pilot_forced.jsonl` | — |
| 6 | `& $PY scripts\stopping_pilot.py fit --train-runs $E\train` | `runs\pilot_models\models.pkl` + SHA-256 | — |
| 7 | `& $PY scripts\stopping_pilot.py score --run pilot_cal` | `runs\pilot_cal\pilot_scores.jsonl` | — |
| 8 | `& $PY scripts\stopping_pilot.py calibrate` | `runs\pilot_calibration\calibration.json` + `.sha256` | instrument passed (refuses otherwise); record the SHA-256 in the run log |
| 9 | `& $PY scripts\prepare_fresh_problems.py --config pilot_eval` | N_E ids, disjoint from pilot_cal | 0 overlap |
| 10 | traces, activations as 3–4 with `pilot_eval` | | |
| 11 | `& $PY scripts\stopping_pilot.py force --run pilot_eval` | refuses unless step 8 exists | — |
| 12 | `& $PY scripts\stopping_pilot.py score --run pilot_eval` | | |
| 13 | `& $PY scripts\stopping_pilot.py evaluate` | `runs\pilot_evaluation\evaluation.json` | — |

Run each step with `2>&1 | Tee-Object -Append $E\logs\<step>.log`.

## 4. Abort, stop and resume

**Abort without reading outcomes** (protocol §3.8) if: any id overlap; the
re-forcing agreement or attrition gate fails (calibrate refuses by itself);
a forced-token count is missing; GPU hours exceed the approved cap; any
plan, grid, threshold or code changes after calibration; or the run would
write into an existing directory (every script refuses).

**To stop** (power, heat, or the cap): press Ctrl+C once in the running
step, or end the python process. Every GPU step writes one row per forced
boundary or per problem and skips finished rows when re-run, so at most the
block in progress is redone. **To resume**, re-run the same command. Never
delete or edit a partial output to "clean up"; if a file is damaged, stop and
record it.

**Never** run `calibrate` after step 11 has started (it refuses), re-run
`fit` after `calibrate` (it refuses to overwrite), or edit
`calibration.json` (evaluation checks its hash).

## 5. CPU dry run of `fit` (2026-10-24)

Run on stored rep1 Stage 1 data with `protocol_TEST_ONLY.json` (placeholder
values, not decisions) into a scratch directory; inputs only read. Result:
see the session record (D98).
