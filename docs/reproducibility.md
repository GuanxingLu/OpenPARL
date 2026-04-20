# Reproducibility

## Hardware

* 1 × H200 (80 GB, SXM) node — 8 GPUs.
* CUDA 12.x, driver 550 or newer.
* ~200 GB disk for MODEL / DATA roots (pre-tokenized + HF checkpoints).

## Software

Pinned via `install.sh`:

* `miles@v0.1-openparl` (= `radixark/miles@5d11fe2f0` + 4 PARL hook commits).
* SGLang, Megatron-LM, Ray — transitive deps of miles.
* Python 3.10+.

## Environment variables

Launchers read these (defaults in each script; `export` to override):

| Variable | Purpose |
|---|---|
| `DATA_ROOT` | Dataset root (wideseek-r1-train, widesearch-test, asearcher-test) |
| `MODEL_ROOT` | HF checkpoint root (Qwen3-4B, Qwen3-0.6B) |
| `REPO_DIR` | OpenPARL checkout (auto-derived from `$(dirname $0)`) |
| `WANDB_API_KEY` | **Required** — no default |
| `WANDB_BASE_URL` | Defaults to `https://api.wandb.ai` |
| `HF_HOME` | HuggingFace cache |
| `OPENPARL_RAG_SERVER` | Host:port of the local RAG server (default `localhost:8000`) |
| `OPENPARL_SUBAGENT_MAX_TURNS` | Per-subagent turn budget (default 8) |
| `OPENPARL_SUBAGENT_MAX_TOOLCALLS` | Per-subagent tool-call budget (default 10) |
| `OPENPARL_SUBAGENT_CONCURRENCY` | Max parallel subagents (default 32) |

## Seeds

All launchers pin seeds. Record the three run seeds used for the blog
numbers here once wandb is pulled.

* **_TBD_** — author to fill.

## Wall-clock

* WideSearch PARL 4B: **_N hours for M steps_** on 1 × H200 node (author to fill).
* Single-agent baseline: **_N'_** (author to fill).

## Data

| Dataset | Path | Purpose |
|---|---|---|
| `wideseek-r1-train/hybrid_20k.miles.jsonl` | `${DATA_ROOT}` | Training |
| `widesearch-test/test.miles.jsonl` | `${DATA_ROOT}` | WideSearch eval |
| `asearcher-test/HotpotQA_rand1000/test.miles.jsonl` | `${DATA_ROOT}` | QA eval |
| `asearcher-test/2WikiMultihopQA_rand1000/test.miles.jsonl` | `${DATA_ROOT}` | QA eval |
| `asearcher-test/Bamboogle/test.miles.jsonl` | `${DATA_ROOT}` | QA eval |

Prepare via `python -m openparl.widesearch.prepare_data`.

## Known issues

* Colocate actor size must be ≤ rollout size for the frozen-subagent
  weight-sync path; otherwise CPU backup racing with Actor pushes causes
  rollout death.
* OOM at `--max-tokens-per-gpu >= 32768`; launcher default (20480) is
  a safe margin for 4B + 0.6B on one H200 (entropy fp32 peak ~5.8 GiB).
* `--disable-entropy-computation` is required for the 4B + 0.6B combo;
  without it, `compute_entropy_from_logits` OOMs at the entropy fp32
  upcast.
