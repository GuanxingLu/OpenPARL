# Architecture

## Decoupled Orchestrator + frozen Subagent

PARL separates agent roles:

* **Orchestrator** (trainable) — receives the user query, decides which
  sub-queries to spawn in parallel, aggregates results, emits the final
  answer. RL gradient flows only on its tokens.
* **Subagent** (frozen) — instantiated from a fixed checkpoint; executes
  one assigned sub-query at a time. Its tokens are **environmental
  observations** to the Orchestrator, not gradient-bearing decision points.

Rationale (paper §3.2): joint optimization over Orchestrator + Subagent
has sparse + ambiguous credit assignment — *"correct final answer ≠
flawless subagent execution; failure ≠ universal subagent error."*
Freezing the Subagent collapses the problem to single-agent RL on the
Orchestrator, conditioned on a stable tool-use environment. The
engineering cost is a separate SGLang engine for the frozen model.

## SGLang deployment (colocate)

OpenPARL runs two SGLang engines in colocate mode:

1. **Orchestrator engine** — weights track the training actor, updated
   each Actor `update_weights` cycle.
2. **Subagent engine** — weights pinned at the frozen Subagent checkpoint,
   CPU-backup enabled so they survive Actor weight pushes unscathed.

The `enable_weights_cpu_backup` configuration lives in the sglang yaml
(`configs/sglang_4B.yaml`). No miles source change is needed for the
weight-sync; the framework already supports per-engine weight isolation.

## Tools

The Orchestrator is wired to three tool classes:

1. **Search** (`search`) — backed by the local RAG server in
   `third_party/rag_server/`.
2. **`create_subagent(name, system_prompt)`** — registers a named subagent
   configuration. Subagent types are *not* hand-defined; they emerge from
   RL-driven specialization.
3. **`assign_task(agent, prompt)`** — dispatches a task to one registered
   subagent. Supports concurrent dispatch: the Orchestrator emits multiple
   tool calls in one turn, and OpenPARL's rollout driver awaits them in
   parallel.

## Critical steps

Episode length is bounded by **critical steps**, not total tool calls:

```
CriticalSteps = Σ_t ( S_main^(t) + max_i S_sub,i^(t) )
```

This directly incentivizes parallel decomposition: spawning N parallel
subagents costs `max_i steps`, not `Σ_i steps` — so balanced splits that
shrink the longest branch reduce the metric.

Critical-step reward weighting is implemented in user-space
(`src/openparl/widesearch/reward.py`) using the `per_token_advantages`
field added to miles `Sample`. The miles side is intentionally agnostic:
whoever populates the field owns the weighting scheme.

## Context management

Agent Swarm as proactive context management (paper §3.4): subagents hold
independent working memories, return only task-relevant outputs to the
Orchestrator. The Orchestrator's context contains high-level coordination
signals + summarized subagent outputs, not full interaction traces.
OpenPARL's `src/openparl/widesearch/orchestrator_tools.py::dispatch`
returns a trimmed summary to the Orchestrator rather than the raw
subagent trajectory.
