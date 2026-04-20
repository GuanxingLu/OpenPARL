"""Widesearch PARL v2 reward.

Same three-component PARL shape as ``math/reward.py``:

  score = r_perf + λ₁ · r_parallel + λ₂ · r_finish

differences vs math:

- ``r_perf`` is rule-based: ``item_f1`` over ``required_columns`` × rows
  aligned by the ``unique_columns`` row-key when the label carries them
  (WideSeek-R1 train / WideSearch eval); falls back to normalized EM on the
  ASearcher QA benchmarks.
- ``LAMBDA_BOX`` is gone (widesearch answers are markdown tables, not
  ``\\boxed{…}``).
- ``PARALLEL_CAP`` drops from 16 to 10 to match the WideSeek-R1 paper's
  best-performing subagent count.

Per-turn advantage decomposition + group-norm is the same as math and
shares ``_fill_per_token_advantages``-shape logic.
"""

from __future__ import annotations

import json

import numpy as np

from miles.utils.types import Sample

from .reward_utils import compute_eval_metrics

LAMBDA1_INIT = 0.3  # r_parallel
LAMBDA2_INIT = 0.2  # r_finish
PARALLEL_CAP = 10
ANNEAL_FRAC = 100.0  # effectively no anneal; flip this when r_perf stops being sparse
GRPO_STD_EPS = 1e-6

_step = 0


def _annealed_lambdas(args) -> tuple[float, float]:
    global _step
    _step += 1
    total = max(1, int(args.num_rollout) * max(1, int(getattr(args, "rollout_batch_size", 1))))
    frac = _step / max(1, int(ANNEAL_FRAC * total))
    scale = max(0.0, 1.0 - frac)
    return LAMBDA1_INIT * scale, LAMBDA2_INIT * scale


def _turn_spans(loss_mask: list[int]) -> list[tuple[int, int]]:
    """Return [start, end) half-open intervals of contiguous loss_mask=1 runs."""
    spans: list[tuple[int, int]] = []
    n = len(loss_mask)
    i = 0
    while i < n:
        if loss_mask[i] == 1:
            j = i
            while j < n and loss_mask[j] == 1:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def _decode_label(label: str | None) -> tuple[str, list[str] | None, list[str] | None]:
    """Return (answer_text, unique_columns, required_columns) from a label.

    prepare_data.py writes every row's label as ``json.dumps({"answer": ...,
    "unique_columns": ... | None, "required_columns": ... | None})``. The
    ``required_columns`` field is optional — older ``.miles.jsonl`` files
    produced before the item-F1 upgrade don't carry it, in which case
    ``item_f1_from_markdown`` defaults to the full GT header. Malformed
    labels fall back to ``(string, None, None)`` so EM still works.
    """
    if not label:
        return "", None, None
    try:
        obj = json.loads(label)
    except (ValueError, TypeError):
        return str(label), None, None
    if isinstance(obj, dict):
        answer = obj.get("answer", "") or ""
        uc = obj.get("unique_columns")
        rc = obj.get("required_columns")
        uc_list = [str(c) for c in uc] if isinstance(uc, (list, tuple)) and uc else None
        rc_list = [str(c) for c in rc] if isinstance(rc, (list, tuple)) and rc else None
        return str(answer), uc_list, rc_list
    return str(obj), None, None


def _score_one(args, sample: Sample, lam1: float, lam2: float) -> dict:
    response = sample.response or ""
    answer, unique_columns, required_columns = _decode_label(sample.label)

    eval_metrics = compute_eval_metrics(response, answer, unique_columns, required_columns)
    # r_perf stays on the same metric as before bbe13569's item-F1 upgrade:
    # item_f1 on widesearch, strict em on QA. The extra cover_em / token_f1 /
    # row_f1 / is_success live only in metadata for eval reporting — training
    # signal is unchanged so checkpoints remain compatible.
    r_perf = eval_metrics["item_f1"] if unique_columns else eval_metrics["em"]

    turns = (sample.metadata or {}).get("turns") or []
    n_assign_total = sum(int(t.get("n_assign", 0)) for t in turns)
    n_valid_total = sum(int(t.get("n_valid", 0)) for t in turns)
    n_create_total = sum(int(t.get("n_create", 0)) for t in turns)

    r_parallel = min(n_assign_total, PARALLEL_CAP) / PARALLEL_CAP
    r_finish = (n_valid_total / n_assign_total) if n_assign_total > 0 else 0.0

    score = r_perf + lam1 * r_parallel + lam2 * r_finish

    per_turn_r: list[float | None] = []
    for t in turns:
        if t.get("final", False):
            per_turn_r.append(None)  # placeholder; filled with r_final downstream
            continue
        n_assign_t = int(t.get("n_assign", 0))
        n_valid_t = int(t.get("n_valid", 0))
        if n_assign_t == 0:
            per_turn_r.append(0.0)
        else:
            per_turn_r.append(lam1 * (n_assign_t / PARALLEL_CAP) + lam2 * (n_valid_t / n_assign_t))
    r_final = r_perf

    critical_steps = int((sample.metadata or {}).get("critical_steps", 0))
    registry_size = int((sample.metadata or {}).get("registry_size", 0))

    if sample.metadata is None:
        sample.metadata = {}
    sample.metadata["raw_reward"] = r_perf
    sample.metadata["eval_metrics"] = eval_metrics

    return {
        "score": score,
        "r_perf": r_perf,
        "r_parallel": r_parallel,
        "r_finish": r_finish,
        "n_assign": n_assign_total,
        "n_create": n_create_total,
        "n_valid": n_valid_total,
        "critical_steps": critical_steps,
        "registry_size": registry_size,
        "lambda1": lam1,
        "lambda2": lam2,
        "has_unique_columns": 1.0 if unique_columns else 0.0,
        "_per_turn_r": per_turn_r,
        "_r_final": r_final,
    }


def _fill_per_token_advantages(samples: list[Sample], score_dicts: list[dict]) -> None:
    """Group-normalized per-turn advantages. Identical logic to math/reward."""
    finals = np.array([sd["_r_final"] for sd in score_dicts], dtype=np.float64)
    non_final_pool = [r for sd in score_dicts for r in sd["_per_turn_r"] if r is not None]

    final_mean = float(finals.mean())
    final_std = float(finals.std())
    final_centered = finals - final_mean
    if final_std > GRPO_STD_EPS:
        final_advs = final_centered / (final_std + GRPO_STD_EPS)
    else:
        final_advs = final_centered

    if non_final_pool:
        nf_mean = float(np.mean(non_final_pool))
        nf_std = float(np.std(non_final_pool))
    else:
        nf_mean = 0.0
        nf_std = 0.0

    for i, sample in enumerate(samples):
        if sample.loss_mask is None or sample.response_length <= 0:
            continue
        spans = _turn_spans(sample.loss_mask)
        if not spans:
            continue
        per_turn_r = score_dicts[i]["_per_turn_r"]

        advs = [0.0] * sample.response_length
        for k, (ss, se) in enumerate(spans):
            if k >= len(per_turn_r):
                raw = nf_mean
                adv = (raw - nf_mean) / (nf_std + GRPO_STD_EPS) if nf_std > GRPO_STD_EPS else 0.0
            elif per_turn_r[k] is None:
                adv = float(final_advs[i])
            else:
                raw = per_turn_r[k]
                adv = (raw - nf_mean) / (nf_std + GRPO_STD_EPS) if nf_std > GRPO_STD_EPS else raw - nf_mean
            for t in range(ss, se):
                advs[t] = float(adv)
        sample.per_token_advantages = advs


async def reward_func(args, sample_or_samples, **kwargs):
    lam1, lam2 = _annealed_lambdas(args)

    if isinstance(sample_or_samples, list):
        samples = sample_or_samples
        dicts = [_score_one(args, s, lam1, lam2) for s in samples]
        _fill_per_token_advantages(samples, dicts)
        return [{k: v for k, v in d.items() if not k.startswith("_")} for d in dicts]

    d = _score_one(args, sample_or_samples, lam1, lam2)
    return {k: v for k, v in d.items() if not k.startswith("_")}
