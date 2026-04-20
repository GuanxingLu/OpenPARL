# Reproducing Kimi K2.5's Agent Swarm with Qwen3-4B

> Draft skeleton — author fills section 5 from wandb runs at blog-writing
> time. Target length: ~2500–3500 words. Publish on X with link back to
> <https://github.com/GuanxingLu/OpenPARL>.

## 1. Why PARL?

_TODO: sequential-agent latency grows super-linearly with task difficulty
(paper Fig 8 shows single-agent WideSearch time ~7× baseline at 70%
Item-F1). Parallel decomposition keeps it near-constant. K2.5's bet: make
the planner itself learn to parallelize via RL._

## 2. Architecture primer

_TODO: decoupled trainable Orchestrator + frozen Subagent; credit-assignment
rationale; why freezing collapses multi-agent back to single-agent RL.
Embed the diagram from `docs/architecture.md`._

## 3. Reproducing on Qwen3-4B + Qwen3-0.6B-frozen

_TODO: scale choices (why 4B + 0.6B is the cheapest cell that still
separates Orchestrator and Subagent capacity), hardware (1 × H200), cost
estimate, training wall-clock._

## 4. The 191 LOC of framework hooks

The PARL recipe needs four small framework hooks in miles. They ship as
four commits on [`GuanxingLu/miles@openparl-v1`](https://github.com/GuanxingLu/miles/tree/openparl-v1):

- `feat(sample): per-token advantages for turn-level credit assignment`
- `feat(args): --disable-entropy-computation flag`
- `feat(metrics): multi-agent pass@k + tool-call-parse-failure + paper-style @k`
- `feat(rollout): allow group_rm during eval for multi-agent rollouts`

_TODO: walk through each commit — what the paper requires, which exact
~50 LOC in miles it touches, why it's the minimal hook. Link to commit
URLs. The point of this section is to show the framework work, not just
the agent code._

## 5. Observations

_TODO (author fills from wandb):_

### 5.1 Training dynamics

_reward curve; critical-step curve; average parallelism — does it match
paper Fig 4's smooth monotonic rise without collapse?_

### 5.2 Emergent subagent specialization

_cluster `create_subagent` calls by their system prompts over training —
do we observe the paper's "Biography Researcher" / "Verification
Specialist" / "File Downloader" archetypes emerging, or different ones?_

### 5.3 Single vs. swarm vs. orchestrator_only on WideSearch

_cover-EM / token-F1 / item-F1 / is_success at the three launchers. How
much of the gap is the Subagent vs. the critical-step budget vs. the
prompt?_

### 5.4 Serial-collapse ablation

_what happens without the critical-step budget? Does the Orchestrator
default to a single subagent and lose the parallel advantage?_

### 5.5 False-tool-call rate

_over training, does the Orchestrator learn to emit fewer malformed
`create_subagent` / `assign_task` calls? This was added as a direct
failure-mode metric._

## 6. What I'd change next

_TODO: curriculum (start with smaller Subagent and scale up during
training); r_parallel ablation (explicit parallel reward vs implicit via
critical-step budget); BrowseComp; In-house Swarm Bench; different
Subagent family (e.g. base model vs. instruct)._

## 7. Code + setup

Repo: <https://github.com/GuanxingLu/OpenPARL>

_TODO: citation blurb (BibTeX for the Kimi paper + this repo); thanks._
