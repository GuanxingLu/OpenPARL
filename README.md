# OpenPARL

**Research reproduction of Kimi K2.5 Agent Swarm (PARL) on WideSearch.**
arXiv:2602.02276, *"Kimi K2.5: Visual Agentic Intelligence"* (Kimi Team, 2026).

> OpenPARL is a personal learning independent reproduction, **not** an official Kimi /
> Moonshot product. No endorsement by the paper authors is implied.

## Features

A minimal, from-scratch reproduction of the K2.5 PARL recipe:

- **Qwen3-4B Orchestrator** trained with RL (GRPO + TIS, icepop) +
  **Qwen3-4B Subagent**, frozen, serving `assign_task` on a separate
  SGLang engine.
- **PARL** follows K2.5 Appendix E.8 literally. **Delegate-only** is a
stricter-than-paper ablation that strips the direct-tool fallback to
probe what happens when the Orchestrator must delegate.

See [**BLOG.md**](BLOG.md) for the full write-up.

## Install

OpenPARL runs inside the official miles container. Nothing is installed
on your host.

```bash
# 1. Clone on the host.
git clone https://github.com/GuanxingLu/OpenPARL.git
cd OpenPARL

# 2. Enter the miles container with OpenPARL mounted.
docker run --gpus all -it --shm-size=32g --privileged \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
    -v "$(pwd)":/workspace/OpenPARL \
    radixark/miles:latest /bin/bash

# 3. Inside the container:
cd /workspace/OpenPARL && ./install.sh
```

## Usage

All commands run inside the miles container. Paths below assume `DATA_ROOT`
and `MODEL_ROOT` — set them to wherever you've staged the assets (defaults
to `./DATA` and `./MODEL` relative to the repo).

### 1. Stage data and model weights

```bash
# Required assets (download / symlink into place).
ls ${DATA_ROOT}/wiki-2018-corpus/{qdrant,wiki_corpus.jsonl,wiki_webpages.jsonl}
ls ${DATA_ROOT}/wideseek-r1-train/hybrid_20k.miles.jsonl
ls ${MODEL_ROOT}/{Qwen3-4B,Qwen3-4B_torch_dist,e5-base-v2}

# If hybrid_20k.miles.jsonl is missing, rebuild the train/eval shards:
python -m openparl.widesearch.prepare_data
```

### 2. Bring up the RAG stack (persistent, survives restarts)

```bash
pip install qdrant-client==1.16.2

# Qdrant vector DB on :6333
chmod +x ${DATA_ROOT}/wiki-2018-corpus/qdrant/qdrant  # one-time
tmux new -d -s qdrant "cd ${DATA_ROOT}/wiki-2018-corpus/qdrant && ./qdrant"

# E5 retrieval server on :8000 (or set PORT=... to avoid collisions)
tmux new -d -s rag "PORT=8000 bash scripts/launch_rag_server.sh; exec bash"
```

Leave both tmux sessions running — they don't need to be restarted between
training runs.

### 3. Train

```bash
export OPENPARL_RAG_SERVER=localhost:8000    # must match the PORT above
export WANDB_API_KEY=...                     # required
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # one H200 node

DATA_ROOT=/path/to/DATA \
MODEL_ROOT=/path/to/MODEL \
FLASHINFER_DISABLE_VERSION_CHECK=1 \
bash scripts/run-qwen3-4B-parl.sh            # PARL          (--agent-mode parl)
# or:
bash scripts/run-qwen3-4B-delegate-only.sh   # Delegate-only (--agent-mode delegate-only)
bash scripts/run-qwen3-4B-single.sh          # Single        (single-agent)
```

Checkpoints land under `saves/Qwen3-4B-<mode>/<RUN_ID>/`.

## RL Infra

The PARL training recipe needs ~191 LOC of hooks in miles. They ship as
4 paper-legible commits on
[`GuanxingLu/miles@openparl-v1`](https://github.com/GuanxingLu/miles/tree/openparl-v1)
(tag `v0.1-openparl`):

| Commit | What it enables |
|---|---|
| `feat(sample): per-token advantages for turn-level credit assignment` | Routes advantage only to Orchestrator tokens; Subagent tokens are zero-grad |
| `feat(args): --disable-entropy-computation flag` | Lets 4B + 4B frozen subagent fit one H200 node (skips the fp32 entropy allocation peak) |
| `feat(metrics): multi-agent pass@k + tool-call-parse-failure + paper-style @k` | Correct `pass_reward` accounting when rollout emits non-primary trajectories; false-tool-call rate; avg@N / max@N aggregators |
| `feat(rollout): allow group_rm during eval for multi-agent rollouts` | Unblocks eval when the reward function sees the whole (Orchestrator + Subagent) group |

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
