"""PARL v2 custom rollout logger (agent-swarm flavor).

Bolts two things onto miles' default wandb logging:

1. Per-component reward stats. ``reward.py::reward_func`` returns a dict
   (``r_perf``, ``r_parallel``, ``r_finish``, ``r_box``, ``n_assign``,
   ``n_create``, ``n_valid``, ``registry_size``, ``critical_steps``,
   ``lambda1/2/_box``) but miles only pipes ``args.reward_key=score`` to
   wandb. We unpack and log mean/std/p50/max/min per key under
   ``reward/<key>/…``.

2. Agent-swarm multi-turn metrics. Each orchestrator turn is a
   "spawn / create / answer" decision; we aggregate from
   ``sample.metadata["turns"]`` and friends:
     - ``assign_rate`` (any turn with n_assign > 0) and per-turn n_assign
       distribution;
     - ``turns_per_rollout`` = len(metadata["turns"]);
     - ``effective_response_ratio`` (loss-mask coverage — drops if subagent
       observation tokens leak into loss);
     - ``n_assign`` / ``r_parallel`` within-group std averaged across prompts
       — the single hardest check that GRPO advantages on dispatch are
       non-degenerate (see plan §"风险与对策");
     - assign-by-difficulty: bucket prompts by group r_perf pass rate and
       report mean ``n_assign`` per bucket — direct proxy for "learned
       scheduling";
     - ``n_unique_agents_used`` per rollout (distinct agent names appearing
       in valid assign_task calls, approximated as ``registry_size``);
     - ``solver_success_rate`` = Σ n_valid / Σ n_assign.

Returns False so miles' default logger still runs after this hook.
"""

from __future__ import annotations

import numpy as np

from miles.utils import tracking_utils
from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import compute_at_k_over_metrics, compute_rollout_step

# Metric names that are 0/1 by construction — pass@k (binary gate on `== 1`)
# is meaningful here. Continuous metrics (item_f1, row_f1, token_f1) only get
# avg@N / max@N. See examples/parl_v2/widesearch/reward_utils.compute_eval_metrics.
_BINARY_EVAL_METRICS = frozenset({"em", "cover_em", "is_success"})

# Keys logged with full 5-stat distribution (mean/std/p50/max/min).
_REWARD_STAT_KEYS = (
    "r_perf",
    "r_parallel",
    "r_finish",
    "r_box",
    "n_assign",
    "n_create",
    "registry_size",
    "critical_steps",
)

# Lambda keys: step-level hyperparams, only mean is meaningful (batch-internal
# std is ~1e-6), so we log a single scalar instead of 5-stat distribution.
_REWARD_SCALAR_KEYS = ("lambda1", "lambda2", "lambda_box")


def _stats(values, prefix):
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}/mean": float(arr.mean()),
        f"{prefix}/std": float(arr.std()),
        f"{prefix}/p50": float(np.median(arr)),
        f"{prefix}/max": float(arr.max()),
        f"{prefix}/min": float(arr.min()),
    }


def _extract(samples, key):
    out = []
    for s in samples:
        r = s.reward
        if isinstance(r, dict) and key in r and r[key] is not None:
            out.append(float(r[key]))
    return out


def _per_turn_field_counts(samples, field: str):
    """Flatten per-turn ``field`` across all samples, for distribution stats."""
    out = []
    for s in samples:
        turns = (s.metadata or {}).get("turns") or []
        for t in turns:
            out.append(float(t.get(field, 0)))
    return out


def _per_turn_assign_counts(samples):
    return _per_turn_field_counts(samples, "n_assign")


def _sample_field_totals(samples, field: str):
    """Per-sample total of ``field`` summed across that sample's turns."""
    out = []
    for s in samples:
        turns = (s.metadata or {}).get("turns") or []
        out.append(float(sum(int(t.get(field, 0)) for t in turns)))
    return out


def _delegate_ratios(samples):
    """Per-sample ``n_assign / (n_assign + n_direct)`` where
    ``n_direct = n_search + n_access``. Samples with no tool usage of any
    kind are skipped (no signal). This is the primary serial-collapse
    diagnostic in swarm-paper mode: if the mean ratio drifts toward 0
    while r_perf still rises, the Orchestrator has learned to bypass
    delegation in favor of direct tools.
    """
    ratios = []
    for s in samples:
        turns = (s.metadata or {}).get("turns") or []
        n_assign = sum(int(t.get("n_assign", 0)) for t in turns)
        n_direct = sum(int(t.get("n_search", 0)) + int(t.get("n_access", 0)) for t in turns)
        denom = n_assign + n_direct
        if denom > 0:
            ratios.append(float(n_assign / denom))
    return ratios


def _compute_reward_component_metrics(samples):
    log_dict = {}
    for key in _REWARD_STAT_KEYS:
        log_dict |= _stats(_extract(samples, key), f"reward/{key}")

    # Lambdas are step-level hyperparams — only log scalar mean.
    for key in _REWARD_SCALAR_KEYS:
        vals = _extract(samples, key)
        if vals:
            log_dict[f"reward/{key}"] = float(np.mean(vals))

    # solver_success_rate = n_valid / n_assign (captures the n_valid info
    # without duplicating the full 5-stat distribution of n_assign).
    valid = sum(_extract(samples, "n_valid"))
    total = sum(_extract(samples, "n_assign"))
    if total > 0:
        log_dict["reward/solver_success_rate"] = float(valid / total)
    return log_dict


def _compute_multi_turn_metrics(args, samples):
    log_dict = {}

    # assign_rate: fraction of samples that spawned at least one subagent.
    # (Full n_assign distribution is already under reward/n_assign/*.)
    n_assign = _extract(samples, "n_assign")
    if n_assign:
        arr = np.asarray(n_assign)
        log_dict["multi_turn/assign_rate"] = float((arr > 0).mean())

    # per-turn n_assign distribution (how many subagents spawned per spawn turn).
    per_turn_assigns = _per_turn_assign_counts(samples)
    if per_turn_assigns:
        log_dict |= _stats(per_turn_assigns, "multi_turn/assign_per_turn")

    # per-turn n_search / n_access distribution (direct-tool parallelism).
    # Non-zero only in swarm-paper / single-agent modes (swarm-strict
    # doesn't expose these tools to the Orchestrator).
    per_turn_search = _per_turn_field_counts(samples, "n_search")
    if any(v > 0 for v in per_turn_search):
        log_dict |= _stats(per_turn_search, "multi_turn/search_per_turn")
    per_turn_access = _per_turn_field_counts(samples, "n_access")
    if any(v > 0 for v in per_turn_access):
        log_dict |= _stats(per_turn_access, "multi_turn/access_per_turn")

    # Per-sample totals — useful to see "how much did this rollout search"
    # independent of turn count. Only emitted when non-trivial.
    n_search_total = _sample_field_totals(samples, "n_search")
    if any(v > 0 for v in n_search_total):
        log_dict |= _stats(n_search_total, "multi_turn/n_search_total")
        # direct_tool_rate: fraction of samples that issued any direct tool call.
        arr = np.asarray(n_search_total) + np.asarray(_sample_field_totals(samples, "n_access"))
        log_dict["multi_turn/direct_tool_rate"] = float((arr > 0).mean())
    n_access_total = _sample_field_totals(samples, "n_access")
    if any(v > 0 for v in n_access_total):
        log_dict |= _stats(n_access_total, "multi_turn/n_access_total")

    # delegate_ratio — serial-collapse diagnostic for swarm-paper mode.
    # Only meaningful when at least some direct-tool capability is present
    # (swarm-strict will always have ratio=1 since n_direct=0).
    delegate_ratios = _delegate_ratios(samples)
    if delegate_ratios and any(v > 0 for v in n_search_total + n_access_total):
        log_dict |= _stats(delegate_ratios, "multi_turn/delegate_ratio")

    # turns_per_rollout — total orchestrator turns actually taken.
    turn_counts = [len((s.metadata or {}).get("turns") or []) for s in samples]
    if turn_counts:
        log_dict |= _stats(turn_counts, "multi_turn/turns_per_rollout")

    # loss-mask coverage: effective / total response length.
    ratios = []
    for s in samples:
        if s.response_length and s.response_length > 0:
            ratios.append(s.effective_response_length / s.response_length)
    if ratios:
        log_dict |= _stats(ratios, "multi_turn/effective_response_ratio")

    # (n_unique_agents_used removed — identical to reward/registry_size/*.)

    # GRPO within-group std: if this is ~0, dispatch gradients collapse.
    if args.advantage_estimator != "ppo":
        groups = group_by(samples, lambda s: s.group_index)
        for key in ("n_assign", "r_parallel", "r_perf"):
            per_group_std = []
            for g in groups.values():
                vals = _extract(g, key)
                if len(vals) >= 2:
                    per_group_std.append(float(np.std(vals)))
            if per_group_std:
                log_dict[f"multi_turn/group_std/{key}/mean"] = float(np.mean(per_group_std))
                log_dict[f"multi_turn/group_std/{key}/frac_nonzero"] = float(
                    np.mean([v > 1e-6 for v in per_group_std])
                )

        # Assign-by-difficulty: bucket prompts by group mean r_perf, report n_assign.
        for label, lo, hi in (("hard", 0.0, 0.25), ("mid", 0.25, 0.75), ("easy", 0.75, 1.01)):
            bucket = []
            for g in groups.values():
                perfs = _extract(g, "r_perf")
                assigns = _extract(g, "n_assign")
                if not perfs or not assigns:
                    continue
                pass_rate = float(np.mean(perfs))
                if lo <= pass_rate < hi:
                    bucket.extend(assigns)
            if bucket:
                log_dict[f"multi_turn/n_assign_by_diff/{label}/mean"] = float(np.mean(bucket))
                log_dict[f"multi_turn/n_assign_by_diff/{label}/count"] = len(bucket)

    return log_dict


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    log_dict = {}
    log_dict |= _compute_reward_component_metrics(samples)
    log_dict |= _compute_multi_turn_metrics(args, samples)

    if log_dict:
        step = compute_rollout_step(args, rollout_id)
        log_dict["rollout/step"] = step
        tracking_utils.log(args, log_dict, step_key="rollout/step")

    # Return False so miles' default _log_rollout_data still runs (we want
    # response_len, truncated_ratio, the scalar rollout/reward, perf/*, etc.).
    return False


def _collect_eval_metric_streams(samples) -> dict[str, list[float]]:
    """Flatten per-sample ``metadata["eval_metrics"]`` into per-metric streams.

    Preserves the input sample order so ``compute_at_k_over_metrics`` can
    reshape to (num_prompts, n_samples_per_eval_prompt) correctly.
    """
    streams: dict[str, list[float]] = {}
    for s in samples:
        md = getattr(s, "metadata", None) or {}
        em = md.get("eval_metrics")
        if not em:
            continue
        for k, v in em.items():
            streams.setdefault(k, []).append(float(v))
    return streams


def _compute_eval_metric_at_k(args, samples) -> dict[str, float]:
    """Paper-aligned Avg@N / Max@N / Pass@k per eval sub-metric.

    Expects per-sample dicts in ``metadata["eval_metrics"]`` (populated by
    ``examples.parl_v2.widesearch.reward.compute_eval_metrics``). Emits
    ``eval/<key>/<metric>/avg@N`` etc. via the outer log_dict.
    """
    streams = _collect_eval_metric_streams(samples)
    group_size = int(getattr(args, "n_samples_per_eval_prompt", 1) or 1)
    return compute_at_k_over_metrics(streams, group_size, _BINARY_EVAL_METRICS)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics):
    log_dict = {}
    for eval_key, payload in data.items():
        samples = payload.get("samples")
        if not samples:
            continue
        log_dict |= {f"eval/{eval_key}/{k}": v for k, v in _compute_reward_component_metrics(samples).items()}
        log_dict |= {f"eval/{eval_key}/{k}": v for k, v in _compute_multi_turn_metrics(args, samples).items()}
        log_dict |= {f"eval/{eval_key}/{k}": v for k, v in _compute_eval_metric_at_k(args, samples).items()}

    if log_dict:
        step = compute_rollout_step(args, rollout_id)
        log_dict["eval/step"] = step
        tracking_utils.log(args, log_dict, step_key="eval/step")

    return False
