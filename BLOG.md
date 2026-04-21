# Reproducing Kimi K2.5's Agent Swarm on Qwen3-4B

Kimi K2.5 (Feb 2026, §3.1) warns about two failure modes when training a
delegating Orchestrator with plain outcome reward:

1. **Serial collapse**: the policy retreats to single-agent execution
   because independent-subagent feedback is sparse / non-stationary.
2. **Spurious parallelism**: the policy fires off arbitrarily many
   subagents to hack intermediate signals.

The paper proposes a decoupled architecture and a
`r_perf + λ₁·r_parallel + λ₂·r_finish` reward to sidestep both. The
paper does **not** give closed-form formulas for `r_parallel` /
`r_finish`, nor the annealing schedule.

I trained three Qwen3-4B RL runs that differ along a single knob:
`--agent-mode`, which bundles the orchestrator system prompt *and*
the exposed tool inventory. Everything else (optimizer, dataset,
critic, rollout budget, GRPO config) is identical.

| Run | `--agent-mode` | Direct tools<br/>(search / browse / python) | Subagent tools<br/>(`create_subagent` / `assign_task`) |
|---|---|:---:|:---:|
| **Single**              | `single-agent` | ✓ | ✗ |
| **Delegate-only** (swarm-strict) | `swarm`        | ✗ | ✓ |
| **PARL** (swarm-paper)  | `swarm-paper`  | ✓ | ✓ |

**PARL** is the paper-canonical Appendix E.8 setting. K2.5's
Orchestrator prompt explicitly enumerates *Search*, *Browser*, *Sub
Agent tools* (`create_subagent` / `assign_task`), and *code execution*
as available tools, i.e. the full core toolset **plus** the two swarm
tools. **Delegate-only** is a stricter-than-paper ablation: we strip
the direct tools away so the Orchestrator *must* delegate, isolating
whether the delegation behavior survives when there is no direct-tool
fallback. The relevant contrast below is between these two: both
expose `assign_task`, but only PARL has anywhere to go when a
delegation fails.

## Subagent dynamics in one view

![subagent dynamics](docs/assets/subagent_dynamics.png)

**(a) Opposite phase transitions; opposite capability flow.** Both
runs that expose `assign_task`, Delegate-only (orange, subagent
tools only) and PARL (teal, subagent tools + direct fallback), move
under the *same* RL signal, but in opposite directions. PARL's
`assign_task` call rate climbs **0.03 → 1.00**; Delegate-only's
decays **0.80 → 0.10**. Single has no `assign_task` tool so its rate
is flat at 0 (purple line, bottom). The dashed lines tell the mirror
*capability* story: on PARL, `reward/solver_success_rate` stays
**flat at 0.40–0.46**, i.e. the *frozen subagent doesn't learn*. On
Delegate-only, where the orchestrator hits increasingly narrow
delegations (those that survive), the remaining assignments get
*better* (**0.42 → 0.72**) even as delegation frequency collapses.

The framing that *didn't* hold up: "prompt drives the phase
transition". PARL and Delegate-only don't differ only in prompt;
they differ in whether the orchestrator has a direct-tool fallback at
all. What the data actually shows is: **stripping the direct-tool
fallback (Delegate-only) collapses RL; keeping the paper-canonical
action space (PARL) lets RL discover *when* to delegate.** This is
an action-space finding, not a prompt finding, and it also reframes
K2.5's serial-collapse warning. The paper worried about reward
sparsity; our runs suggest the deeper issue is that a must-delegate
action space gives RL no off-ramp.

**(b) Collapsed policy also degenerates.** Delegate-only's collapse
isn't a harmless retreat to "do it all yourself"; it has no direct
tools to retreat *to*. Truncation rate climbs to **23%** and
repetition to **24%**; PARL stays under **5%** on both, and Single
stays under **3%** (faint purple lines at the bottom). The policy
slowly gives up on producing coherent outputs entirely.

**(c) PARL run: plans widen, deepen, and specialize.** Each
evaluated episode eventually composes **~12 `assign_task` calls
× ~11 `create_subagent` instantiations**, with critical-path length
growing proportionally. The registry of **distinct subagent names**
per episode grows **0 → ~5**, a scalar proxy for K2.5's Figure 6
specialization word-cloud (Biography Researcher / Fact Checker / …)
that our rollout logs don't capture directly. The gap between
`create_subagent` calls (~11) and distinct names (~5) means the
Orchestrator **re-uses** specialist roles across subtasks rather than
spawning fresh ones each time. The Orchestrator is not just calling
`assign_task` more often: it composes larger, deeper, and more
heterogeneous plans, then *re-uses* the specialists it has created.

The delegation dynamics reproduce cleanly, which is the point of this
post. The **target task itself** (WideSearch item-F1) does not
improve in any of the three runs within the 80-rollout window
plotted, and `is_success = 1[item-F1 = 1.0] ≡ 0` throughout. Two
closable gaps explain this and are documented rather than figured so
the dynamics plot stays focused:

- `rollout_max_critical_steps = 48` vs K2.5 Appendix E.8's **100
  orchestrator + 100 subagent** budget for WideSearch;
- `src/openparl/widesearch/reward.py` ships `ANNEAL_FRAC = 100.0`, so
  `λ₁` and `λ₂` **never anneal** under the planned rollout count.
  Constant `r_parallel + r_finish` pressure is what panel (c) is
  really a portrait of: the PARL policy hacks spawn count
  (`n_assign` → ~12/episode, `r_parallel` → 0.70) under the frozen
  schedule. Flipping `ANNEAL_FRAC` and widening the critical-step
  budget are single-line changes in the reward / launcher and are the
  obvious next run.

## Held-out multi-hop QA

![per-task eval grid](docs/assets/per_task_eval_grid.png)

Columns (left → right): HotpotQA, 2WikiMultihop, Bamboogle. Rows (top
→ bottom): pass@1, pass@2, pass@4. EM is solid, cover-EM is dashed.

- **PARL (teal) is clearly ahead on strict EM** across all three
  benchmarks and all three pass levels. The ordering does not invert
  as K grows, so this is not a pass@4 artifact; it's a reproducible
  behavioral effect.
- **cover-EM narrows the gap.** All three runs land in similar
  cover-EM bands, suggesting that subagents already "know" the answer.
  What the Orchestrator learns in the swarm-paper action space is to
  **assemble and format** the final response precisely enough to
  match strict EM.
- **Delegate-only underperforms Single on HotpotQA / 2Wiki cover-EM**,
  another symptom of the behavioral collapse visible in panel (b)
  above.

---

**Runs**: All ran on 1× H200 × 8, Qwen3-4B Orchestrator + Qwen3-4B frozen Subagent, GRPO, ~80 updates.

**Reference:** Kimi Team, *Kimi K2.5: Visual Agentic Intelligence*,
arXiv:2602.02276, §3.1 and Appendix E.8.
