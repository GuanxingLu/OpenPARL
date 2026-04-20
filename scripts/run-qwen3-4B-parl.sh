#!/bin/bash

# PARL v2 widesearch prod run on Qwen3-4B (H200x8).
# Prereq: local RAG server running on :8000. See widesearch/README.md.

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
# Targeted python kill so we don't clobber the long-running RAG server
# (examples/agent/tools/search_local_server_qdrant/local_retrieval_server.py).
# Matches ray workers + miles train.py + parl_v2 launcher processes only.
pkill -9 -f 'ray::\|train\.py\|parl_v2\|run_parl_v2' || true
sleep 3
pkill -9 ray
pkill -9 -f 'ray::\|train\.py\|parl_v2\|run_parl_v2' || true

set -ex

export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Strip any inherited proxy vars; httpx picks them up and tries SOCKS on hosts
# that don't have socksio installed, which kills the RAG client path.
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
NVLINK_COUNT=$(nvidia-smi | grep -o "NVLink" | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

export WANDB_API_KEY=${WANDB_API_KEY:-local-82cbbacfe8e3c0c527da528160bd76a1e85c9fea}
export WANDB_BASE_URL=${WANDB_BASE_URL:-http://33.180.4.104}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../../.." &>/dev/null && pwd)"

DEV_REPO_DIR=${DEV_REPO_DIR:-${REPO_DIR}}
DATA_ROOT=${DATA_ROOT:-${DEV_REPO_DIR}/DATA}
MODEL_ROOT=${MODEL_ROOT:-${DEV_REPO_DIR}/MODEL}
MODE=${MODE:-normal}
NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
RUN_ID=${RUN_ID:-"run_$(date +%Y%m%d_%H%M%S)"}

# widesearch-specific env vars consumed by widesearch/assign_task.py.
export MILES_PARL_V2_RAG_SERVER=${MILES_PARL_V2_RAG_SERVER:-localhost:8000}
export MILES_PARL_V2_SUBAGENT_MAX_TURNS=${MILES_PARL_V2_SUBAGENT_MAX_TURNS:-8}
export MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS=${MILES_PARL_V2_SUBAGENT_MAX_TOOLCALLS:-10}
export MILES_PARL_V2_SUBAGENT_CONCURRENCY=${MILES_PARL_V2_SUBAGENT_CONCURRENCY:-32}

MODEL_ARGS=(
   --env widesearch
   --model qwen3-4B
   --hf-checkpoint "${MODEL_ROOT}/Qwen3-4B"
   --ref-load "${MODEL_ROOT}/Qwen3-4B_torch_dist"
)

RUN_ARGS=(
   --mode "${MODE}"
   --run-id "${RUN_ID}"
   --dev-repo-dir "${DEV_REPO_DIR}"
   --save-path "${DEV_REPO_DIR}/saves/Qwen3-4B-parl-v2-widesearch/${RUN_ID}"
   --rollout-batch-size 64
   # global-batch-size = rollout_batch_size * n_samples_per_prompt (64*8=512)
   # => 1 grad step per rollout (matches RLinf n_minibatches=1, fully on-policy).
   --global-batch-size 512
   --rollout-max-response-len 28672
   --rollout-max-critical-steps 48
   --entropy-coef 0
)

PARALLEL_ARGS=(
   --num-gpus-per-node "${NUM_GPUS}"
   --tensor-model-parallel-size 2
   --rollout-num-gpus-per-engine 2
)

DATA_ARGS=(
   --prompt-data "${DATA_ROOT}/wideseek-r1-train/hybrid_20k.miles.jsonl"
)

GENERATE_ARGS=(
   --generate-max-turns 10
)

EVAL_EXTRA_ARGS=(
   --eval-interval 20
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 28672
   --eval-max-context-len 32768
   --eval-top-p 1
   --log-passrate
   --eval-prompt-data
   widesearch "${DATA_ROOT}/widesearch-test/test.miles.jsonl"
   hotpotqa   "${DATA_ROOT}/asearcher-test/HotpotQA_rand1000/test.miles.jsonl"
   2wiki      "${DATA_ROOT}/asearcher-test/2WikiMultihopQA_rand1000/test.miles.jsonl"
   bamboogle  "${DATA_ROOT}/asearcher-test/Bamboogle/test.miles.jsonl"
)

# Dynamic batch sizing caps per-rank micro-batch tokens. 20480 is chosen to
# fit the fp32 entropy spike in compute_entropy_from_logits: on Qwen3-4B +
# TP=2 the per-rank logits tensor is [N, 75968]; entropy's fp32 upcast
# needs ~N*75968*4 bytes per allocation, and 32768 OOM'd at 8.91 GiB while
# only 7.25 GiB was free. 20480 keeps that peak at ~5.8 GiB. Sequences
# longer than 20480 get split across micro-batches by dynamic batch.
PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

# Override hardcoded optimizer defaults in run_parl_v2.py (weight_decay=0.1,
# adam_beta2=0.98). Passed via --extra-args so argparse last-wins picks these
# up without touching the shared launcher (keeps math runs untouched).
OPTIM_OVERRIDE_ARGS=(
   --weight-decay 0.01
   --adam-beta2 0.999
   # Skip entropy in policy loss (entropy-coef is 0 anyway): saves the fp32
   # peak alloc inside compute_entropy_from_logits. See miles/utils/arguments.py.
   --disable-entropy-computation
)

cd "${REPO_DIR}"

RAY_GCS_PORT=${RAY_GCS_PORT:-26379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-28265}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} \
   --port ${RAY_GCS_PORT} \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT}
export RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
export MILES_SCRIPT_EXTERNAL_RAY=1

# See math/run-qwen3-4B-parl-v2.sh for SUBAGENT_MODE semantics (frozen vs shared).
SUBAGENT_MODE=${SUBAGENT_MODE:-frozen}
if [ "$SUBAGENT_MODE" = "frozen" ]; then
   SGLANG_EXTRA_ARGS=(--sglang-config examples/parl_v2/sglang_config_4B.yaml)
elif [ "$SUBAGENT_MODE" = "shared" ]; then
   SGLANG_EXTRA_ARGS=()
else
   echo "ERROR: SUBAGENT_MODE must be 'frozen' or 'shared', got '$SUBAGENT_MODE'" >&2
   exit 1
fi

python examples/parl_v2/run_parl_v2.py \
   ${MODEL_ARGS[@]} \
   ${RUN_ARGS[@]} \
   ${PARALLEL_ARGS[@]} \
   ${DATA_ARGS[@]} \
   ${GENERATE_ARGS[@]} \
   "${SGLANG_EXTRA_ARGS[@]}" \
   --extra-args "${EVAL_EXTRA_ARGS[*]} ${PERF_ARGS[*]} ${OPTIM_OVERRIDE_ARGS[*]}"
