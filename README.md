# OpenPARL

**Research reproduction of Kimi K2.5 Agent Swarm (PARL) on WideSearch.**
arXiv:2602.02276 — *"Kimi K2.5: Visual Agentic Intelligence"* (Kimi Team, 2026).

> OpenPARL is an independent reproduction, **not** an official Kimi / Moonshot
> product. No endorsement by the paper authors is implied.

## What it does

Trains a **Qwen3-4B Orchestrator** with RL on the WideSearch benchmark while a
**Qwen3-0.6B Subagent** stays frozen. The Orchestrator uses `create_subagent`
and `assign_task` tools to dispatch parallel sub-queries; per-token credit
assignment routes RL advantages only to Orchestrator tokens (Subagent tokens
are treated as environmental observations, not differentiable decision
points).

Three launcher configurations are provided for comparison:

| Launcher | Agent mode | Purpose |
|---|---|---|
| `scripts/run-qwen3-4B-parl.sh` | Trainable Orchestrator + frozen Subagent | Headline result |
| `scripts/run-qwen3-4B-single.sh` | Single agent (no subagents) | Baseline |
| `scripts/run-qwen3-4B-orchestrator_only.sh` | Orchestrator without subagents, paper-aligned prompt | Ablation |

## Install

OpenPARL runs inside the official miles container. Nothing is installed
on your host.

```bash
# 1. Clone on the host
git clone https://github.com/GuanxingLu/OpenPARL.git
cd OpenPARL

# 2. Enter the miles container with OpenPARL mounted
docker run --gpus all -it --shm-size=32g --privileged \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:65536 \
    -v "$(pwd)":/workspace/OpenPARL \
    radixark/miles:latest /bin/bash

# 3. Inside the container:
cd /workspace/OpenPARL && ./install.sh
```

`install.sh` overlays 4 PARL framework hook commits on top of the image's
miles (via `pip install --no-deps ...@v0.1-openparl`, leaving sglang /
megatron / ray untouched) and installs OpenPARL in editable mode.

For the exact miles image tag used for the blog results, see
[`docs/reproducibility.md`](docs/reproducibility.md).

## Reproduce

```bash
# 1. Launch local RAG server on :8000
bash scripts/launch_rag_server.sh

# 2. Run the headline launcher (Qwen3-4B + frozen Qwen3-0.6B subagent)
bash scripts/run-qwen3-4B-parl.sh
```

See [`docs/reproducibility.md`](docs/reproducibility.md) for hardware,
environment variables, seeds, and expected wall-clock.

## Results

*(Populate from wandb runs at blog-writing time.)*

| Config | item-F1 | row-F1 | is_success | avg@N | max@N | pass@N |
|---|---|---|---|---|---|---|
| Single | — | — | — | — | — | — |
| Orchestrator-only | — | — | — | — | — | — |
| PARL (swarm) | — | — | — | — | — | — |

## Repository map

```
src/openparl/             agent code (prompts, generate, rollout_log, run, tool)
  widesearch/             widesearch-specific (reward, prompts, tools, prepare-data)
third_party/rag_server/   RAG server vendored from RLinf
configs/                  sglang configs (4B + 0.6B)
scripts/                  launchers (.sh)
tests/                    CPU-only unit tests
docs/                     architecture / reward / reproducibility
```

## Framework hooks

The PARL training recipe needs ~191 LOC of hooks in miles. They ship as 4
paper-legible commits on
[`GuanxingLu/miles@openparl-v1`](https://github.com/GuanxingLu/miles/tree/openparl-v1)
(tag `v0.1-openparl`):

| Commit | What it enables |
|---|---|
| `feat(sample): per-token advantages for turn-level credit assignment` | Routes advantage only to Orchestrator tokens; Subagent tokens are zero-grad |
| `feat(args): --disable-entropy-computation flag` | Lets 4B + frozen 0.6B fit into one H200 node (skips the fp32 entropy allocation peak) |
| `feat(metrics): multi-agent pass@k + tool-call-parse-failure + paper-style @k` | Correct `pass_reward` accounting when rollout emits non-primary trajectories; false-tool-call rate; avg@N / max@N aggregators |
| `feat(rollout): allow group_rm during eval for multi-agent rollouts` | Unblocks eval when the reward model needs the whole (Orchestrator + Subagent) group |

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Cite

If you use OpenPARL, please cite the Kimi K2.5 paper and this repository.
