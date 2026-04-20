# Reward

```
score = r_perf  +  λ₁ · r_parallel  +  λ₂ · r_finish
```

## `r_perf` — task-level outcome

Rule-based, varies by eval set:

* **WideSearch / WideSeek-R1 train**: `item_f1` over `required_columns` ×
  rows, aligned by the `unique_columns` row-key. URL + multi-value cell
  canonicalization is applied on both sides before comparison.
* **ASearcher QA** (HotpotQA / 2Wiki / Bamboogle): normalized exact-match
  on the boxed answer (searches both `\boxed{…}` and `<answer>…</answer>`
  blocks).

Implementation: `src/openparl/widesearch/reward.py` and
`src/openparl/widesearch/reward_utils.py`.

## `r_finish` — sub-agent completion rate

Fraction of `assign_task` calls that returned a parseable, schema-valid
result. Discourages **spurious parallelism** (spawning many nonsense
subagents to inflate the parallel term).

## `r_parallel` — disabled by default

Paper's original three-term formula was designed to counter **serial
collapse** (Orchestrator defaulting to a single subagent). The explicit
formula is not given in the paper.

OpenPARL's defaults in `reward.py`:
- `LAMBDA1_INIT = 0.3` (r_parallel)
- `LAMBDA2_INIT = 0.2` (r_finish)
- `ANNEAL_FRAC = 100.0` — effectively no annealing; flip this when
  `r_perf` stops being sparse.

The critical-step budget already implicitly rewards parallelism (fixed
budget + more parallel subagents → more total reasoning → higher r_perf),
so `r_parallel` is often redundant and can be disabled by setting its λ
to 0.

## Annealing

Paper specifies λ₁, λ₂ anneal to 0 over training so the final policy
optimizes `r_perf` alone. The schedule shape (linear / cosine / step) is
not given — OpenPARL currently uses `ANNEAL_FRAC = 100.0` (equivalent to
"don't anneal"); tune if serial collapse stops being a concern.
