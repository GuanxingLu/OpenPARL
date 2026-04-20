# Reproducing Kimi K2.5's Agent Swarm on Qwen3-4B

Kimi K2.5 (Feb 2026, §3.1) warns about two failure modes when training a
delegating Orchestrator with plain outcome reward:

1. **Serial collapse** — the policy retreats to single-agent execution
   because independent-subagent feedback is sparse / non-stationary.
2. **Spurious parallelism** — the policy fires off arbitrarily many
   subagents to hack intermediate signals.

The paper proposes a decoupled architecture and a
`r_perf + λ₁·r_parallel + λ₂·r_finish` reward to sidestep both. The
paper does **not** give closed-form formulas for `r_parallel` /
`r_finish`, nor the annealing schedule.

I trained three Qwen3-4B RL runs that differ **only** in the
orchestrator prompt + which tools are exposed. Everything else —
optimizer, dataset, critic, rollout budget, GRPO config — is identical:

| Run | Orchestrator prompt | Tools available |
|---|---|---|
| `single-baseline` | single-agent prompt | search / browse / python |
| `orch-only`       | **default** prompt  | search / browse / python / `create_subagent` / `assign_task` |
| `paper-config`    | K2.5 orchestrator prompt | search / browse / python / `create_subagent` / `assign_task` |

## Behavior, schedule, and target task in one view

![phase transition](docs/assets/phase_transition.png)

**(a) Two opposite phase transitions, same tool set.** Both runs expose
`create_subagent` + `assign_task`. With the K2.5 orchestrator prompt
(teal), `assign_task` call rate climbs **0.03 → 1.00**. With the
default prompt (orange), it decays **0.80 → 0.10** — RL actively
*unlearns* delegation. This operationalizes K2.5's theoretical
serial-collapse as a sub-100-step phase transition, driven by prompt
rather than by the policy's tool inventory.

**(b) Collapsed policy also degenerates.** It isn't a harmless
single-agent fallback. Truncation rate climbs to **23%** and
repetition to **24%** on the default-prompt run; the paper-config run
stays under **5%** on both. The policy slowly gives up on producing
coherent outputs.

**(c) Paper run: plans widen and deepen.** Each evaluated episode
eventually composes **~12 `assign_task` calls × ~11 `create_subagent`
instantiations**, with critical-path length growing proportionally.
The Orchestrator is not just calling `assign_task` more often — it
composes larger, deeper plans to spread across frozen subagents.

**(d) Target task does not improve.** WideSearch item-F1 (the
training reward's widesearch term) stays flat-to-down across all three
runs, and `is_success = 1[item-F1 = 1.0] ≡ 0` the whole way. Two
closable gaps explain this:

- `rollout_max_critical_steps = 48` vs the paper's Appendix E.8 budget
  of **100 orchestrator + 100 subagent** steps for WideSearch;
- `examples/parl_v2/widesearch/reward.py` ships
  `ANNEAL_FRAC = 100.0`, so `λ₁` and `λ₂` **never anneal** under the
  planned rollout count. Pressure from `r_parallel + r_finish` stays
  constant, and the policy meets it by hacking spawn count (`n_assign`
  climbs to ~12/episode, `r_parallel` saturates at 0.70) while item-F1
  on WideSearch actually drops **0.059 → 0.048**.

The delegation dynamics reproduce cleanly. Closing the WideSearch gap
needs a correct annealing schedule and a matching step budget — both
are single-line changes in the launcher / reward file.

## Held-out multi-hop QA — ordering is stable across pass@K

![per-task eval grid](docs/assets/per_task_eval_grid.png)

EM (solid) and cover-EM (dashed) on HotpotQA, 2WikiMultihop, and
Bamboogle at pass@{1, 2, 4}.

- **paper-config (teal) is clearly ahead on strict EM** across all
  three benchmarks and all three pass levels. The ordering does not
  invert as K grows, so this is not a pass@4 artifact — it's a
  reproducible behavioral effect.
- **cover-EM narrows the gap.** All three runs land in similar
  cover-EM bands, suggesting that subagents already "know" the answer.
  What the orchestrator learns under the paper prompt is to
  **assemble and format** the final response precisely enough to match
  strict EM.
- **The default-prompt run underperforms single-agent on HotpotQA /
  2Wiki cover-EM** — another symptom of the behavioral collapse
  visible in panel (b) above.

---

**Runs**: All ran on 1× H200 × 8, Qwen3-4B
Orchestrator + Qwen3-4B frozen Subagent, GRPO, ~80 updates.
Reproduction launchers:
[`scripts/run-qwen3-4B-{single,orchestrator_only,parl}.sh`](scripts/).

**Reference:** Kimi Team, *Kimi K2.5: Visual Agentic Intelligence*,
arXiv:2602.02276, §3.1 and Appendix E.8.
