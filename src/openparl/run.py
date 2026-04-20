"""PARL v2 launcher (retool_v2 style).

Wraps train.py with the PARL v2 args (orchestrator with `create_subagent` +
`assign_task` agent-swarm tools). Mirrors examples/retool_v2/run_retool_multi_turn.py —
the .sh launchers in this folder are thin shells that invoke this script.

NOTE on PYTHONPATH: parl_v2 only lives in the dev tree
(/workspace/miles/examples/...), not in the baked-in /root/miles. To make
`import examples.parl_v2.*` resolve correctly we pass an absolute
train_script under the dev tree — `python3 <abs>/train.py` puts that
directory at sys.path[0], shadowing both /root/miles and Megatron's own
`examples` package.
"""

import os
import socket
import subprocess
import time
from dataclasses import dataclass, field
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

WANDB_PROJECT = "miles-dev-multi-agent"

DEFAULT_DEV_REPO_DIR = "/workspace/miles"

_MODEL_DEFAULTS = {
    "qwen3-4B": {
        "hf_checkpoint": "MODEL/Qwen3-4B",
        "ref_load": "MODEL/Qwen3-4B_torch_dist",
        "megatron_model_type": "qwen3-4B",
        "tensor_model_parallel_size": 2,
        "rollout_num_gpus_per_engine": 2,
    },
    "qwen3-0.6B": {
        "hf_checkpoint": "MODEL/Qwen3-0.6B",
        "ref_load": "MODEL/Qwen3-0.6B_torch_dist",
        "megatron_model_type": "qwen3-0.6B",
        "tensor_model_parallel_size": 1,
        "rollout_num_gpus_per_engine": 1,
    },
    "qwen3-30B-A3B": {
        "hf_checkpoint": "MODEL/Qwen3-30B-A3B",
        "ref_load": "MODEL/Qwen3-30B-A3B_torch_dist",
        "megatron_model_type": "qwen3-30B-A3B",
        "tensor_model_parallel_size": 4,
        "rollout_num_gpus_per_engine": 4,
    },
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "normal"
    run_id: str = field(default_factory=U.create_run_id)
    hardware: Literal["H100", "GB200", "GB300"] = "H100"
    num_gpus_per_node: int | None = None
    model: Literal["qwen3-4B", "qwen3-0.6B", "qwen3-30B-A3B"] = "qwen3-0.6B"
    # Selects the environment-specific reward / assign_task implementation.
    # Each env lives under examples/parl_v2/<env>/ with reward.py + assign_task.py.
    env: Literal["math", "widesearch"] = "math"
    dev_repo_dir: str = DEFAULT_DEV_REPO_DIR
    save_path: str = ""
    prompt_data: str = ""
    eval_prompt_data: str = ""
    generate_max_turns: int = 6
    rollout_max_context_len: int = 32768
    rollout_max_response_len: int = 4096
    # K2.5 PARL episode-length budget in TURN units (not tokens):
    #   phase_cost = 1 per orchestrator turn (final / create-only / length)
    #              = 2 per spawn turn (≥1 executed assign_task, max_i S_sub=1
    #                under single-shot subagents).
    # Defaults to 2 * generate_max_turns (loose cap; main turn cap is
    # --generate-max-turns).
    rollout_max_critical_steps: int = 0
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    global_batch_size: int = 64
    num_rollout: int = 500
    entropy_coef: float = 0.001
    sglang_router_ip: str = "127.0.0.1"
    sglang_router_port: int = 18765
    sglang_router_prometheus_port: int = 14444
    # empty string means "use default for the selected model"
    hf_checkpoint: str = ""
    ref_load: str = ""
    megatron_model_type: str = ""
    tensor_model_parallel_size: int = 0
    rollout_num_gpus_per_engine: int = 0
    # Optional path to a miles --sglang-config YAML. When set, miles carves
    # the rollout pool into multiple SGLang models (used for the frozen
    # subagent topology). Empty -> miles default single-model single-pool.
    sglang_config: str = ""
    # Three-way control over the Orchestrator tool surface:
    #   swarm        : current default — Orchestrator holds only
    #                  ``create_subagent`` + ``assign_task``. Delegation is
    #                  architecturally forced.
    #   swarm-paper  : paper-faithful — Orchestrator additionally holds
    #                  direct ``search`` / ``access``. Delegation becomes
    #                  a learned capability, motivated by context sharding.
    #                  Widesearch-only (math has no direct tools).
    #   single-agent : widesearch → Orchestrator holds only direct
    #                  ``search`` / ``access`` (no subagent). Math →
    #                  strips the entire PARL v2 layer and runs plain
    #                  single-turn GRPO against rm-type=deepscaler (the
    #                  isolated-variable baseline for parl_v2/math).
    agent_mode: Literal["swarm", "swarm-paper", "single-agent"] = "swarm"
    extra_args: str = ""

    def __post_init__(self):
        self.num_gpus_per_node = self.num_gpus_per_node or U.NUM_GPUS_OF_HARDWARE[self.hardware]
        defaults = _MODEL_DEFAULTS[self.model]
        self.hf_checkpoint = self.hf_checkpoint or f"{self.dev_repo_dir}/{defaults['hf_checkpoint']}"
        self.ref_load = self.ref_load or f"{self.dev_repo_dir}/{defaults['ref_load']}"
        self.megatron_model_type = self.megatron_model_type or defaults["megatron_model_type"]
        self.tensor_model_parallel_size = self.tensor_model_parallel_size or defaults["tensor_model_parallel_size"]
        self.rollout_num_gpus_per_engine = self.rollout_num_gpus_per_engine or defaults["rollout_num_gpus_per_engine"]
        if self.env == "math" and self.agent_mode == "swarm-paper":
            raise ValueError(
                "agent_mode=swarm-paper is widesearch-only: math has no direct " "Orchestrator-side tools to expose."
            )
        if self.env == "math":
            self.prompt_data = self.prompt_data or f"{self.dev_repo_dir}/DATA/dapo-math-17k/dapo-math-17k.jsonl"
            self.eval_prompt_data = self.eval_prompt_data or f"{self.dev_repo_dir}/DATA/aime-2024/aime-2024.jsonl"
        elif self.env == "widesearch":
            self.prompt_data = self.prompt_data or f"{self.dev_repo_dir}/DATA/wideseek-r1-train/hybrid_20k.miles.jsonl"
            # widesearch eval launchers pass multi-set --eval-prompt-data via --extra-args;
            # leave this empty by default so run_parl_v2 skips its single-set expansion.
        self.rollout_max_critical_steps = self.rollout_max_critical_steps or (2 * self.generate_max_turns)
        if not self.save_path:
            suffix_map = {
                "swarm": "parl-v2",
                "swarm-paper": "parl-v2-paper",
                "single-agent": "baseline",
            }
            suffix = suffix_map[self.agent_mode]
            self.save_path = f"{self.dev_repo_dir}/saves/{os.path.basename(self.hf_checkpoint)}-{suffix}/{self.run_id}"


_WANDB_VARIANT_BY_MODE = {
    "swarm": "parl-v2",
    "swarm-paper": "parl-v2-paper",
    "single-agent": "baseline",
}


def _get_wandb_args(args: ScriptArgs) -> str:
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
    variant = _WANDB_VARIANT_BY_MODE[args.agent_mode]
    return (
        "--use-wandb "
        f"--wandb-project {WANDB_PROJECT} "
        f"--wandb-group {args.model}-{variant}-{args.env} "
        f"--wandb-key {WANDB_API_KEY} "
    )


def prepare(args: ScriptArgs):
    hf_dir = os.path.dirname(args.hf_checkpoint)
    U.convert_checkpoint(
        model_name=os.path.basename(args.hf_checkpoint),
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        hf_checkpoint=args.hf_checkpoint,
        dir_dst=hf_dir,
    )


_TOOL_SPECS_PATH = {
    # Swarm-strict is literally examples.parl_v2.tool.tool_specs for both
    # envs; widesearch swarm-paper / single-agent re-compose via the
    # widesearch/orchestrator_tools.py module (which also carries the
    # direct dispatch coroutine).
    ("math", "swarm"): "examples.parl_v2.tool.tool_specs",
    ("widesearch", "swarm"): "examples.parl_v2.tool.tool_specs",
    ("widesearch", "swarm-paper"): "examples.parl_v2.widesearch.orchestrator_tools.tool_specs_swarm_paper",
    ("widesearch", "single-agent"): "examples.parl_v2.widesearch.orchestrator_tools.tool_specs_single",
}

_ORCHESTRATOR_PROMPT_PATH = {
    # swarm mode leaves this empty — generate.py falls back to the
    # ORCHESTRATOR_SYSTEM_PROMPT constant at runtime.
    "swarm": "",
    "swarm-paper": "examples.parl_v2.prompts.ORCHESTRATOR_SYSTEM_PROMPT_PAPER",
    "single-agent": "examples.parl_v2.prompts.ORCHESTRATOR_SYSTEM_PROMPT_SINGLE",
}

# Env-specific direct-tool dispatchers. Math has none (no direct tools).
_DIRECT_TOOLS_PATH = {
    "widesearch": "examples.parl_v2.widesearch.orchestrator_tools.dispatch",
}


def execute(args: ScriptArgs):
    megatron_model_type = args.megatron_model_type

    ckpt_args = (
        f"--hf-checkpoint {args.hf_checkpoint} "
        f"--ref-load {args.ref_load} "
        f"--load {args.save_path} "
        f"--save {args.save_path} "
        f"--save-interval {2 if args.mode == 'debug_minimal' else 50} "
    )

    is_math_single = args.env == "math" and args.agent_mode == "single-agent"

    if is_math_single:
        # Baseline: no orchestrator, no tools, no group reward — plain
        # single-turn GRPO against miles' built-in deepscaler rm. Relies on
        # --apply-chat-template so miles' default generate wraps the raw
        # prompt string with the model's chat template (parl_v2 path does
        # this itself inside examples.parl_v2.generate).
        custom_args = "--rm-type deepscaler " "--apply-chat-template "
    else:
        tool_specs_path = _TOOL_SPECS_PATH[(args.env, args.agent_mode)]
        custom_args = (
            "--custom-generate-function-path examples.parl_v2.generate.generate "
            f"--generate-tool-specs-path {tool_specs_path} "
            "--generate-tool-call-parser qwen25 "
            f"--generate-max-turns {args.generate_max_turns} "
            f"--assign-task-impl-path examples.parl_v2.{args.env}.assign_task.call "
            "--log-multi-turn "
            f"--custom-rm-path examples.parl_v2.{args.env}.reward.reward_func "
            "--custom-rollout-log-function-path examples.parl_v2.rollout_log.log_rollout_data "
            "--custom-eval-rollout-log-function-path examples.parl_v2.rollout_log.log_eval_rollout_data "
            # --group-rm hands the full rollout group to reward_func, which is
            # required so it can group-normalize per-turn rewards and populate
            # sample.per_token_advantages for K2.5-style turn-level credit.
            "--group-rm "
        )
        prompt_path = _ORCHESTRATOR_PROMPT_PATH[args.agent_mode]
        if prompt_path:
            custom_args += f"--orchestrator-prompt-path {prompt_path} "
        # Direct-tool dispatcher only makes sense when the Orchestrator
        # actually holds direct tools, i.e., swarm-paper or single-agent.
        if args.agent_mode in ("swarm-paper", "single-agent") and args.env in _DIRECT_TOOLS_PATH:
            custom_args += f"--orchestrator-direct-tools-path {_DIRECT_TOOLS_PATH[args.env]} "

    rollout_args = (
        f"--prompt-data {args.prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--rollout-shuffle "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-context-len {args.rollout_max_context_len} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {args.global_batch_size} "
        "--balance-data "
        f"--sglang-router-ip {args.sglang_router_ip} "
        f"--sglang-router-port {args.sglang_router_port} "
    )
    if not is_math_single:
        # --reward-key selects which field parl_v2.reward.reward_func writes
        # into sample.reward; --rollout-max-critical-steps is the K2.5
        # turn-budget cap. Both apply to every mode that goes through the
        # parl_v2 custom path (including widesearch single-agent) — only
        # the math deepscaler branch opts out.
        rollout_args += "--reward-key score " f"--rollout-max-critical-steps {args.rollout_max_critical_steps} "

    eval_args = ""
    if args.mode != "debug_minimal":
        # math's legacy single-set eval; widesearch passes its own multi-set
        # --eval-prompt-data via --extra-args (and leaves eval_prompt_data empty).
        eval_prompt_flag = f"--eval-prompt-data aime {args.eval_prompt_data} " if args.eval_prompt_data else ""
        eval_args = (
            "--eval-interval 20 "
            # "--skip-eval-before-train "
            f"{eval_prompt_flag}"
            "--n-samples-per-eval-prompt 4 "
            f"--eval-max-response-len {args.rollout_max_response_len} "
            f"--eval-max-context-len {args.rollout_max_context_len} "
            "--eval-top-p 1 "
            "--log-passrate "
        )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        f"--entropy-coef {args.entropy_coef} "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    if not is_math_single:
        # icepop TIS is parl_v2's default multi-turn-friendly correction.
        # The math deepscaler baseline is single-turn GRPO, so leaving TIS
        # off there avoids confounding multi-agent vs single-agent with a
        # second variable. Widesearch single-agent is still multi-turn
        # (search → access → answer), so it keeps TIS like its swarm peers.
        grpo_args += "--use-tis " "--custom-tis-function-path miles.backends.training_utils.loss.icepop_function "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.rollout_num_gpus_per_engine} " "--sglang-mem-fraction-static 0.7 "
    )
    if args.sglang_config:
        sglang_args += f"--sglang-config {args.sglang_config} "

    # Perf defaults mirror examples/retool_v2 and scripts/run-qwen3-4B.sh.
    # --sequence-parallel halves LayerNorm/Dropout activation memory when TP>1
    # and is required to keep long agentic rollouts from OOMing during loss
    # (logits.clone() in ppo_utils.calculate_log_probs_and_entropy). Recompute
    # flags match every miles 4B RL launcher — omitting them kept all 36 Qwen
    # layer activations live through backward. Launcher --extra-args can still
    # override any of these (argparse last-wins), which is how the 30B-A3B
    # parl_v2 launcher bumps --expert-model-parallel-size to 8.
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_model_parallel_size} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{_get_wandb_args(args)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{custom_args} "
        f"{args.extra_args} "
    )

    # The assign_task tool needs to call back into SGLang during a
    # rollout, so it must know the router address ahead of time. miles'
    # _start_router skips its own launch when --sglang-router-ip is set, so
    # we pre-launch the router ourselves on the same host:port. This must
    # happen AFTER execute_train's `pkill -9 sglang` cleanup phase but
    # BEFORE the ray job is submitted — the before_ray_job_submit hook
    # gives us exactly that window.
    def _launch_router():
        log_dir = f"{args.dev_repo_dir}/logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = f"{log_dir}/sglang_router.log"
        with open(log_path, "ab") as log_f:
            proc = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "sglang_router.launch_router",
                    "--host",
                    args.sglang_router_ip,
                    "--port",
                    str(args.sglang_router_port),
                    "--prometheus-port",
                    str(args.sglang_router_prometheus_port),
                    "--log-level",
                    "warn",
                ],
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # Brief wait for the port to bind so engine /workers POSTs succeed.
        for _ in range(30):
            if proc.poll() is not None:
                raise RuntimeError(f"sglang_router exited early (rc={proc.returncode}); see {log_path}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((args.sglang_router_ip, args.sglang_router_port)) == 0:
                    print(
                        f"sglang_router pid={proc.pid} listening on "
                        f"{args.sglang_router_ip}:{args.sglang_router_port}"
                    )
                    return
            time.sleep(1)
        raise RuntimeError(f"sglang_router pid={proc.pid} did not bind {args.sglang_router_port} in time")

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=megatron_model_type,
        # Absolute dev-tree train.py: ensures sys.path[0]=/workspace/miles so
        # `examples.parl_v2.*` imports resolve to the dev copy and not
        # the baked-in /root/miles or Megatron's own examples package.
        train_script=f"{args.dev_repo_dir}/train.py",
        before_ray_job_submit=_launch_router,
        extra_env_vars={
            "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
            "MILES_SGLANG_ROUTER_IP": args.sglang_router_ip,
            "MILES_SGLANG_ROUTER_PORT": str(args.sglang_router_port),
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            # Forward wandb config to worker-node actors. WANDB_API_KEY is
            # also passed via --wandb-key, but WANDB_BASE_URL has no CLI
            # equivalent — without forwarding it, actors on remote nodes
            # default to wandb.ai and time out in egress-restricted envs.
            **{k: os.environ[k] for k in ("WANDB_BASE_URL", "WANDB_API_KEY") if k in os.environ},
            # Multi-node NCCL transport tuning (mirrors
            # scripts/run-glm4.5-355B-A32B.sh). Without these, the first
            # cross-node NCCL collective — all_gather_object inside
            # Megatron's _get_param_groups, right after param count is
            # printed — fails with ncclRemoteError on Connect.
            # NCCL_SOCKET_IFNAME, GLOO_SOCKET_IFNAME, MASTER_ADDR,
            # no_proxy and NCCL_NVLS_ENABLE are already handled by
            # miles/utils/external_utils/command_utils.py.
            # NCCL_IB_HCA selects the RoCE HCAs (mlx5_bond_0..7 on this
            # cluster — prefix match picks all 8). Uncomment NCCL_DEBUG
            # to re-diagnose cross-node transport if it regresses.
            **{k: os.environ[k] for k in ("TP_SOCKET_IFNAME",) if k in os.environ},
            "NCCL_IB_HCA": "mlx5_bond",
            "NCCL_CUMEM_ENABLE": "0",
            "NVTE_BWD_LAYERNORM_SM_MARGIN": "20",
            "NCCL_IB_TC": "160",
            "NCCL_PXN_DISABLE": "0",
            "NCCL_IB_GID_INDEX": "3",
            "NCCL_NET_GDR_LEVEL": "4",
            "NCCL_IB_RETRY_CNT": "7",
            "NCCL_IB_TIMEOUT": "32",
            "NCCL_IB_QPS_PER_CONNECTION": "8",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_MIN_CTAS": "4",
            # "NCCL_DEBUG": "INFO",
        },
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
