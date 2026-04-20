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

I trained three Qwen3-4B RL runs that differ along a single knob:
`--agent-mode`, which bundles the orchestrator system prompt *and*
the exposed tool inventory. Everything else — optimizer, dataset,
critic, rollout budget, GRPO config — is identical.

| Run | `--agent-mode` | Direct tools<br/>(search / browse / python) | Subagent tools<br/>(`create_subagent` / `assign_task`) |
|---|---|:---:|:---:|
| **Single**              | `single-agent` | ✓ | ✗ |
| **PARL** (swarm-strict) | `swarm`        | ✗ | ✓ |
| **Orchestrator-only** (swarm-paper) | `swarm-paper`  | ✓ | ✓ |

*PARL* is the paper's strict reading — the Orchestrator *must*
delegate, it has no direct tools. *Orchestrator-only* is the
paper-faithful Appendix E.8 setting — the Orchestrator *can* delegate
but keeps direct tools as a fallback. The relevant contrast below is
between these two: both expose `assign_task`, but only
Orchestrator-only has anywhere to go when a delegation fails.

## Behavior, schedule, and target task in one view

![phase transition](docs/assets/phase_transition.png)

**(a) Opposite phase transitions; opposite capability flow.** Both
runs that expose `assign_task` — PARL (orange, subagent tools only)
and Orchestrator-only (teal, subagent tools + direct fallback) — move
under the *same* RL signal, but in opposite directions.
Orchestrator-only's `assign_task` call rate climbs **0.03 → 1.00**;
PARL's decays **0.80 → 0.10**. Single has no `assign_task` tool so
its rate is flat at 0 (purple line, bottom). The dashed lines tell
the mirror *capability* story: on Orchestrator-only,
`reward/solver_success_rate` stays **flat at 0.40–0.46** — the
*frozen subagent doesn't learn*. On PARL, where the orchestrator hits
increasingly narrow delegations (those that survive), the remaining
assignments get *better* — **0.42 → 0.72** — even as delegation
frequency collapses.

The framing that *didn't* hold up: "prompt drives the phase
transition". PARL and Orchestrator-only don't differ only in prompt
— they differ in whether the orchestrator has a direct-tool fallback
at all. What the data actually shows is: **giving the orchestrator
nowhere to go when delegation fails (PARL) collapses RL; giving it a
fallback (Orchestrator-only) lets RL discover *when* to delegate.**
This is an action-space finding, not a prompt finding, and it also
reframes K2.5's serial-collapse warning — the paper worried about
reward sparsity; our runs suggest the deeper issue is that a
must-delegate action space gives RL no off-ramp.

**(b) Collapsed policy also degenerates.** PARL's collapse isn't a
harmless retreat to "do it all yourself" — it has no direct tools to
retreat *to*. Truncation rate climbs to **23%** and repetition to
**24%**; Orchestrator-only stays under **5%** on both, and Single
stays under **3%** (faint purple lines at the bottom). The policy
slowly gives up on producing coherent outputs entirely.

**(c) Orchestrator-only run: plans widen, deepen, and specialize.**
Each evaluated episode eventually composes **~12 `assign_task` calls
× ~11 `create_subagent` instantiations**, with critical-path length
growing proportionally. The registry of **distinct subagent names**
per episode grows **0 → ~5** — a scalar proxy for K2.5's Figure 6
specialization word-cloud (Biography Researcher / Fact Checker / …)
that our rollout logs don't capture directly. The gap between
`create_subagent` calls (~11) and distinct names (~5) means the
Orchestrator **re-uses** specialist roles across subtasks rather than
spawning fresh ones each time. The Orchestrator is not just calling
`assign_task` more often — it composes larger, deeper, and more
heterogeneous plans, then *re-uses* the specialists it has created.

**(d) Target task does not improve.** WideSearch item-F1 (the
training reward's widesearch term) stays flat-to-down across all three
runs, and `is_success = 1[item-F1 = 1.0] ≡ 0` the whole way. Two
closable gaps explain this:

- `rollout_max_critical_steps = 48` vs the paper's Appendix E.8 budget
  of **100 orchestrator + 100 subagent** steps for WideSearch;
- `src/openparl/widesearch/reward.py` ships `ANNEAL_FRAC = 100.0`, so
  `λ₁` and `λ₂` **never anneal** under the planned rollout count.
  Pressure from `r_parallel + r_finish` stays constant, and the
  Orchestrator-only policy meets it by hacking spawn count (`n_assign`
  climbs to ~12/episode, `r_parallel` saturates at 0.70) while item-F1
  on WideSearch actually drops **0.059 → 0.048**. (PARL can't hack
  this way — it's busy degenerating, per panel (b).)

The delegation dynamics reproduce cleanly. Closing the WideSearch gap
needs a correct annealing schedule and a matching step budget — both
are single-line changes in the launcher / reward file.

## Held-out multi-hop QA — ordering is stable across pass@K

![per-task eval grid](docs/assets/per_task_eval_grid.png)

EM (solid) and cover-EM (dashed) on HotpotQA, 2WikiMultihop, and
Bamboogle at pass@{1, 2, 4}.

- **Orchestrator-only (teal) is clearly ahead on strict EM** across
  all three benchmarks and all three pass levels. The ordering does
  not invert as K grows, so this is not a pass@4 artifact — it's a
  reproducible behavioral effect.
- **cover-EM narrows the gap.** All three runs land in similar
  cover-EM bands, suggesting that subagents already "know" the answer.
  What the Orchestrator learns in the swarm-paper action space is to
  **assemble and format** the final response precisely enough to
  match strict EM.
- **PARL underperforms Single on HotpotQA / 2Wiki cover-EM** —
  another symptom of the behavioral collapse visible in panel (b)
  above.

---

**Runs**: All ran on 1× H200 × 8, Qwen3-4B Orchestrator + Qwen3-4B
frozen Subagent, GRPO, ~80 updates. Reproduction launchers map
one-to-one with the arms above:
`scripts/run-qwen3-4B-single.sh` → **Single**,
`scripts/run-qwen3-4B-parl.sh` → **PARL** (swarm-strict),
`scripts/run-qwen3-4B-orchestrator_only.sh` → **Orchestrator-only**
(swarm-paper).

**Reference:** Kimi Team, *Kimi K2.5: Visual Agentic Intelligence*,
arXiv:2602.02276, §3.1 and Appendix E.8.
