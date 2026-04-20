# Two opposite phase transitions under the same RL signal

*Reproducing Kimi K2.5's Agent Swarm on Qwen3-4B — a short note.*

Kimi K2.5 (Feb 2026, §3.1) warns about two failure modes when training a
delegating Orchestrator with plain outcome reward:

1. **Serial collapse** — the policy retreats to single-agent execution
   because independent-subagent feedback is sparse/non-stationary.
2. **Spurious parallelism** — the policy fires off arbitrarily many
   subagents to hack intermediate signals.

The paper proposes a decoupled architecture and a `r_perf + λ₁·r_parallel
+ λ₂·r_finish` reward to sidestep both. The paper does **not** give
closed-form formulas for `r_parallel` / `r_finish`, nor the annealing
schedule.

I trained three Qwen3-4B RL runs that differ **only** in the orchestrator
prompt + which tools are exposed. Everything else — optimizer, dataset,
critic, rollout budget, GRPO config — is identical:

| Run | Orchestrator prompt | Tools available |
|---|---|---|
| `single-baseline` | single-agent prompt | search / browse / python |
| `orch-only`       | **default** prompt  | search / browse / python / `create_subagent` / `assign_task` |
| `paper-config`    | K2.5 orchestrator prompt | search / browse / python / `create_subagent` / `assign_task` |

Both paper failure modes show up — one per run — inside 100 steps.

![phase transition](docs/assets/phase_transition.png)

### (a) Serial collapse vs. emergent delegation

Same tool set. Different system prompt. Opposite phase transitions.

- **paper-config** (green): `assign_task` rate climbs **0.03 → 1.00**.
- **orch-only** (red): `assign_task` rate decays **0.80 → 0.10**. Starts
  willing to delegate (because the tools are there), but RL actively
  *unlearns* delegation under the default prompt.

This makes K2.5's theoretical serial-collapse an observable, <100-step
phase transition — driven by prompt, not by the policy's tool
inventory.

### (b) Paper config: a clean three-way hand-off

In the paper config the three delegation statistics move in lockstep:
`assign_rate` and `delegate_ratio` go to 1.0, `direct_tool_rate` drops
from 0.79 to 0.02. The Orchestrator essentially stops calling tools
itself and becomes a pure dispatcher.

### (c) Orch-only also degenerates behaviorally

The serial-collapse run doesn't just give up delegation — it also
decoheres. Truncation rate climbs from ~0 to **23%**, repetition from
~0 to **24%**, past step 60. The paper run stays under 5% on both. So
the failure mode isn't "harmless single-agent fallback"; it's a policy
slowly giving up on producing coherent outputs.

### One gotcha worth writing down

Training `r_perf/mean` rises **0.13 → 0.23** on the paper run, which is
tempting to read as "the Orchestrator is solving WideSearch better."
It isn't. The 20k training mixture folds WideSearch (reward =
`item_f1`) together with multi-hop QA (reward = strict EM). What's
actually rising is QA-EM; the WideSearch `item_f1` on held-out eval is
**flat-to-declining** (`0.059 → 0.048`), and `is_success = [item_f1 ==
1.0]` stays at **0** across all three runs — partly because of the
reward mixture, partly because our `rollout_max_critical_steps = 48`
versus the paper's 100+100 budget for WideSearch (Appendix E.8).

That is: **the paper's delegation dynamics reproduce cleanly at 4B. The
paper's WideSearch Item-F1 does not — and the failure is already visible
in the reward decomposition, not downstream.** Fixing it needs a proper
`λ₁/λ₂` anneal schedule and a larger critical-step budget, which is
what I'm running next.

---

**Runs**: `pa9lipn3` (single), `tqzr8z9x` (orch-only), `gbamfgd3`
(paper-config). All ran on 1× H200 × 8, Qwen3-4B Orchestrator +
Qwen3-4B Subagent, GRPO, ~80 updates. Reproduction launchers:
[`scripts/run-qwen3-4B-{single,orchestrator_only,parl}.sh`](scripts/).

**Reference:** Kimi Team, *Kimi K2.5: Visual Agentic Intelligence*,
arXiv:2602.02276, §3.1 and Appendix E.8.
