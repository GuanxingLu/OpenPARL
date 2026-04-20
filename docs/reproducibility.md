# Reproducibility

Everything below matches the three runs plotted in [`BLOG.md`](../BLOG.md).

## Hardware

- 1 × H200 node, 8 × SXM 80 GB GPUs.
- CUDA 12.x, driver ≥ 550, NVLink recommended (the launcher auto-detects).
- ~200 GB disk for `${DATA_ROOT}` (prepared `*.miles.jsonl`) +
  `${MODEL_ROOT}` (HF checkpoints + Megatron torch_dist mirror).
- Host does not need Python / CUDA — all runtime comes from the miles
  container.

## Software

OpenPARL runs inside the official miles Docker image. Host only needs
Docker with `--gpus` support; Python / CUDA / SGLang / Megatron / Ray
are baked in.

- Container image: **`radixark/miles:latest`** (Docker Hub, public).
  OpenPARL tracks the latest miles image; if you want bitwise
  reproducibility, record your own `docker image ls --digests` line
  alongside your wandb run.
- Overlaid inside the container by `install.sh`:
  - `miles` fork at tag `v0.1-openparl` = `radixark/miles@5d11fe2f0` +
    4 PARL hook commits (~191 LOC total). Installed with `--no-deps
    --force-reinstall` so the image's sglang / megatron / ray versions
    stay untouched.
  - OpenPARL itself in editable mode (`pip install -e .`).

Verify with:

```bash
python -c "import miles, openparl; print(miles.__version__, openparl.__file__)"
```

## Environment variables

Each launcher reads these from the environment (defaults in the script
itself; `export` to override before invoking):

| Variable | Default | Purpose |
|---|---|---|
| `DATA_ROOT` | `${REPO_DIR}/DATA` | Prepared dataset root |
| `MODEL_ROOT` | `${REPO_DIR}/MODEL` | HF + Megatron checkpoint root |
| `DEV_REPO_DIR` | `${REPO_DIR}` | Checkpoint save prefix |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | GPUs to claim |
| `WANDB_API_KEY` | *(required)* | No default; launcher errors out if unset |
| `WANDB_BASE_URL` | `https://api.wandb.ai` | Override for self-hosted wandb |
| `OPENPARL_RAG_SERVER` | `localhost:8000` | RAG server host:port |
| `OPENPARL_SUBAGENT_MAX_TURNS` | 8 | Per-subagent turn budget |
| `OPENPARL_SUBAGENT_MAX_TOOLCALLS` | 10 | Per-subagent tool-call budget |
| `OPENPARL_SUBAGENT_CONCURRENCY` | 32 | Max simultaneous in-flight subagents |
| `SUBAGENT_MODE` | `frozen` | `frozen` = separate SGLang engine; `shared` = single pool |
| `RUN_ID` | `run_<timestamp>` | Checkpoint subdir |
| `HF_HOME` | *(container default)* | HuggingFace cache |

## Data

```
${DATA_ROOT}/
├── wideseek-r1-train/
│   └── hybrid_20k.miles.jsonl                    # training (WideSearch + multi-hop QA mix, ~20k rows)
├── widesearch-test/
│   └── test.miles.jsonl                          # WideSearch eval
└── asearcher-test/
    ├── HotpotQA_rand1000/test.miles.jsonl        # QA eval
    ├── 2WikiMultihopQA_rand1000/test.miles.jsonl # QA eval
    └── Bamboogle/test.miles.jsonl                # QA eval
```

Build the `.miles.jsonl` files from the upstream HF datasets with:

```bash
python -m openparl.widesearch.prepare_data
# or rebuild:
python -m openparl.widesearch.prepare_data --force
```

`prepare_data.py` assumes the raw datasets are already under `DATA/`;
fetch from HuggingFace separately.

## Models

```
${MODEL_ROOT}/
├── Qwen3-4B/                                     # HF checkpoint (tokenizer + safetensors)
└── Qwen3-4B_torch_dist/                          # Megatron torch_dist mirror (for --ref-load)
```

The `*_torch_dist` mirrors are produced by miles's checkpoint conversion
tools; consult `third_party/miles/tools/` if you need to regenerate.

## Run IDs (blog reference)

The three runs plotted in [`BLOG.md`](../BLOG.md):

| Blog label | Launcher | Wandb run ID | State at snapshot | Duration |
|---|---|---|---|---|
| **PARL**          | `scripts/run-qwen3-4B-orchestrator_only.sh` | `gbamfgd3` | running, ~80 rollouts | ~21.7 h |
| **Delegate-only** | `scripts/run-qwen3-4B-parl.sh`              | `tqzr8z9x` | running, ~94 rollouts | ~21.7 h |
| **Single**        | `scripts/run-qwen3-4B-single.sh`            | `pa9lipn3` | crashed at rollout 119 | ~9.6 h  |

All three ran on `miles-dev-multi-agent` (self-hosted wandb).
Replace `miles-dev-multi-agent` with your own wandb project when
reproducing; the blog plot is regenerated from parquet caches of these
three runs by `docs/assets/make_plots.py` (not committed — regenerate
locally if you need to tweak the figure).

## Wall-clock

One-H200-node figures at `--rollout-max-critical-steps 48`,
`--rollout-batch-size 64`, `--n-samples-per-prompt 8`:

| Run | Rollouts observed | Avg step time (end of training) |
|---|---|---|
| **PARL**          | 80  | ~1600 s/step (delegation adds rollout latency) |
| **Delegate-only** | 94  | ~1400 s/step |
| **Single**        | 119 | ~300 s/step (no subagent dispatch; crashed mid-run) |

Step time grows ~6× over training for the swarm runs as delegation
depth increases. If you are budget-constrained, plan for ~24 h/run at
80 rollouts to reproduce the BLOG plot.

## Known issues

- **`ANNEAL_FRAC = 100.0` in `src/openparl/widesearch/reward.py`
  disables λ₁/λ₂ annealing.** The PARL run shows the classic
  spurious-parallelism signature under this setting
  (`eval/widesearch/reward/n_assign/mean` climbs 0.06 → 12.66 while
  `item_f1` declines). Flip to something ≤ 1.0 to enable annealing if
  you're pushing past the phase-transition experiments.
- **`is_success = [item_f1 == 1.0]` is too strict at 4B + 48 critical
  steps.** All three blog runs report `is_success/pass@4 == 0`. Add a
  lenient `is_success_at_threshold(t)` in
  `src/openparl/widesearch/reward_utils.py` for mid-training signal,
  and raise `--rollout-max-critical-steps` toward the paper's budget
  (100+100) for the final headline number.
- **Training `r_perf` mixes WideSearch `item_f1` with QA strict EM.**
  The 20k training mixture folds both; the `r_perf` key gets the
  appropriate metric per sample. An overall-`r_perf`-mean upswing
  can hide a WideSearch-only decline. Use the per-dataset
  `eval/widesearch/reward/r_perf/mean` channel for the real signal.
- **Colocate actor size must be ≤ rollout size** for the
  frozen-subagent weight-sync path, otherwise CPU backup racing with
  the Actor's weight push kills the rollout engine.
- **OOM at `--max-tokens-per-gpu >= 32768`** on 4B + frozen-4B on one
  H200. Launcher default (`20480`) leaves a safe margin for the fp32
  entropy spike (~5.8 GiB peak vs 7.25 GiB free).
- **`--disable-entropy-computation` is required** for the 4B + 4B
  combo; without it `compute_entropy_from_logits` OOMs at the fp32
  upcast. Set `--entropy-coef 0` in the launcher to keep loss math
  correct.
- **Targeted pkill pattern in launchers.** All three launchers start
  with `pkill -9 -f 'ray::\|train\.py\|openparl'` to avoid clobbering
  a long-running RAG server on the same host. If you are running
  anything else matching that pattern, tune the regex.
