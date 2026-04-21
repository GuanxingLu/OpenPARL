# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

OpenPARL is a from-scratch reproduction of Kimi K2.5's PARL agent-swarm recipe on the WideSearch benchmark. It is a *thin* launcher on top of the [`miles`](https://github.com/GuanxingLu/miles) RL framework (installed separately via `install.sh` from tag `v0.1-openparl`, which carries 4 required framework hooks). All training code paths are reached through `python -m openparl.run`; the `scripts/run-*.sh` shells are thin env-var wrappers around that entrypoint.

## Runtime environment

Everything runs inside the `radixark/miles:latest` Docker container (mounted at `/workspace/OpenPARL`). Do not attempt to install or run training outside the container — `miles` pulls specific pinned versions of `sglang`/`megatron`/`ray` that `install.sh` deliberately preserves via `pip install --no-deps --force-reinstall`.

Training needs two long-running services on `DATA_ROOT/wiki-2018-corpus`:
- Qdrant on `:6333` (binary shipped in the corpus dir)
- E5 retrieval server on `:8000` via `scripts/launch_rag_server.sh`

Leave both running across training launches. The launcher kills `sglang`/`ray`/`openparl` processes at startup with targeted `pkill -f` patterns that deliberately avoid the RAG server — do **not** broaden those to `pkill -9 python`.

## Common commands

```bash
# Install inside the miles container (run once)
./install.sh

# Rebuild *.miles.jsonl training/eval shards from raw DATA/
python -m openparl.widesearch.prepare_data           # use --force to rebuild

# Launch training (all require WANDB_API_KEY + running RAG server)
bash scripts/run-qwen3-4B-parl.sh                    # --agent-mode parl
bash scripts/run-qwen3-4B-delegate-only.sh           # --agent-mode delegate-only
bash scripts/run-qwen3-4B-single.sh                  # --agent-mode single-agent

# Lint (pre-commit config pins ruff/black/isort/autoflake; run via pre-commit)
pre-commit run --all-files
```

There is no test suite (no `tests/` dir, no `pytest.ini`). `pyproject.toml` lists only `typer` + `numpy` as deps because the heavy stack comes from the miles container image.

## The three-launcher axis

The only variable that distinguishes the three launchers is `--agent-mode`, which bundles a tool-spec set *and* an Orchestrator system prompt. Keep this coupled:

| `--agent-mode`  | Tool specs (`_TOOL_SPECS_PATH` in `run.py`)                          | System prompt                                         |
|-----------------|----------------------------------------------------------------------|-------------------------------------------------------|
| `parl`          | `openparl.widesearch.orchestrator_tools.tool_specs_parl`             | `openparl.prompts.ORCHESTRATOR_SYSTEM_PROMPT_PARL`    |
| `delegate-only` | `openparl.tool.tool_specs`                                           | (empty → `ORCHESTRATOR_SYSTEM_PROMPT_DELEGATE_ONLY`)  |
| `single-agent`  | `openparl.widesearch.orchestrator_tools.tool_specs_single`           | `openparl.prompts.ORCHESTRATOR_SYSTEM_PROMPT_SINGLE`  |

`parl` and `single-agent` additionally set `--orchestrator-direct-tools-path` to `openparl.widesearch.orchestrator_tools.dispatch`; `delegate-only` leaves it unset, which is the flag `generate.py` uses to shut off the direct-tool code path entirely.

`prompts.py` duplicates **numeric budgets** (8-name registry cap, 10-tool-call subagent budget, ~5000-char access cap) into the Orchestrator prompts. These numbers are the source of truth in `tool.py` (`MAX_REGISTRY_SIZE`) and `widesearch/assign_task.py` (`OPENPARL_SUBAGENT_*` env defaults). **Keep prompt text in sync with those files when either changes.**

## Architecture: two-level rollout

`generate.py`'s `generate()` is registered via `--custom-generate-function-path` and replaces miles' default multi-turn loop. Each rollout sample is one **Orchestrator** episode:

1. Orchestrator generates → parsed as tool calls by `qwen25` parser.
2. Tool calls are split into **three** phases in `_execute_tool_calls_parallel`:
   - `create_subagent`: sync, inline, pure registry write (no inference).
   - `assign_task`: async, parallel up to `MAX_CONCURRENT_ASSIGN=8`; each call is a full **subagent ReAct loop** in `widesearch/assign_task.py` (multi-turn `search`/`access` against the RAG server, must emit `<result>…</result>`).
   - direct tools (`search`/`access`, PARL & single-agent only): async, parallel up to `MAX_CONCURRENT_DIRECT=16`, routed via `orchestrator_tools.dispatch`.
3. Tool results are appended as `role=tool` messages; iterate up to `--generate-max-turns`.

The subagent hits a **separate** SGLang engine pool (configured in `configs/sglang_4B.yaml`: one `actor` pool with weight updates + one `subagent` pool frozen at the init checkpoint). The miles fork routes `get_model_url(args, "subagent")` to that frozen pool; when `--sglang-config` is omitted, subagent inference falls back to the live actor (shared-weights ablation).

The SGLang **router** is pre-launched by `run.py::_launch_router` in `before_ray_job_submit` — miles otherwise tries to start it after `assign_task` might already be calling into it.

## Critical-steps accounting

`rollout_max_critical_steps` is a **turn-level** budget (not tokens). Per turn, `critical_steps += 1 + (max_i S_sub,i if n_assign>0 else 0)`, where `S_sub,i` is the ReAct step count of the i-th subagent. This mirrors K2.5's "phase cost": 1 per normal Orchestrator turn, 1 + max-subagent-depth per spawn turn. Rollout halts (sample status `TRUNCATED`) when the budget is exceeded. Default in `run.py` is `2 * generate_max_turns`; launchers set it explicitly (PARL = 48).

## Reward → per-token advantages

`widesearch/reward.py::reward_func` is registered via `--custom-rm-path` with `--group-rm`, so it sees the whole GRPO group at once and is responsible for *both* returning the scalar score and filling `sample.per_token_advantages`.

- **Score:** `r_perf + λ₁·r_parallel + λ₂·r_finish`. `r_perf` = item-F1 for widesearch (uses `unique_columns`/`required_columns` from the label), otherwise strict EM for QA. `r_parallel = min(n_assign, 10)/10`. `r_finish = n_valid/n_assign` where `n_valid` requires the subagent produced a `<result>…</result>` block *and* used ≥1 tool.
- **Per-turn credit:** `_fill_per_token_advantages` uses `sample.loss_mask` to recover contiguous turn spans (Orchestrator-only, because miles' `--per-token-advantages` framework hook zeroes Subagent tokens). Non-final turns get a pool-normalized `λ₁/λ₂`-based advantage; the final turn gets the group-normalized `r_perf`.
- **Anneal:** `ANNEAL_FRAC = 100.0` currently disables the λ schedule (see BLOG.md for why — this is the obvious next knob to flip).

`reward_func` also populates `sample.metadata["raw_reward"]` and `eval_metrics` so the rollout logger (`rollout_log.py`) can stratify by component.

## Things that will bite you

- **Absolute `train_script` path.** `run.py` passes `f"{dev_repo_dir}/train.py"` explicitly so `sys.path[0]` points at this checkout, not `/root/miles`. Keep it absolute.
- **Proxy env vars.** Launchers `unset http_proxy https_proxy all_proxy` because `httpx` otherwise tries SOCKS on hosts without `socksio`, killing the RAG client. Don't re-export them.
- **`--no-deps` reinstall of miles.** `install.sh` uses `--no-deps --force-reinstall` to overlay *only* the 4 PARL hook commits without disturbing the container's pinned sglang/megatron/ray. Never drop `--no-deps`.
- **`third_party/rag_server/`** is vendored from RLinf (Apache-2.0). It is pre-commit-excluded and lint-excluded in `pyproject.toml`; don't reformat it.
- **`MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1`** is required — OpenPARL's `generate()` signature matches the refactored rollout API only.
- **`tools/` is gitignored.** `tools/make_plots.py` / `tools/fetch_wandb.py` are local-only helpers for figure generation; they are not part of the reproduction path.
- **Saved checkpoints** land under `${DEV_REPO_DIR}/saves/{ckpt-basename}-{agent_mode}/{run_id}/` (constructed in `ScriptArgs.__post_init__` unless the launcher overrides `--save-path`).
