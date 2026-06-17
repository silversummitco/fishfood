---
name: workload-capacity-planner
description: Plan and reason about throughput, bottlenecks, and completion-time predictability for a batch of work processed by a heterogeneous pool of agents (a few high-capacity + many small workers). Use when someone asks "how many workers do we need", "will we finish the batch on time / reliably", "where's the bottleneck", "should we add more workers or route differently", or wants to size a mixed human/AI agent fleet. Built on the Fish Food model (fish_food.py). Includes a fit-triage gate that says when NOT to use this approach.
---

# Workload Capacity & Predictability Planner

This skill turns a workload + agent-pool description into a capacity/bottleneck
analysis and a **completion-time predictability** estimate, using the Fish Food
sandbox (`fish_food.py`) as its compute engine. Its job is *planning and
insight*, not speed magic and not correctness.

Work through the steps in order. **Do not skip Step 1** — it is what keeps this
honest.

## Step 1 — Fit triage (mandatory gate)

Decide whether the workload is even in scope. Score these:

GOOD fit (all/most true):
- Work is a **batch of discrete, mostly-independent units**.
- A **heterogeneous pool of agents** does the work (a few high-capacity + many
  small/cheap ones).
- Units vary in difficulty, and **hard units require capable agents** (the small
  ones can't finish them).
- **Discovery/assignment has a real cost** — agents must find or be routed to
  work; there is no instant perfect dispatcher.
- The stakeholder cares about the **completion-time distribution**
  (predictability), bottlenecks, and worker-mix tradeoffs.

POOR fit (if any are central, STOP and say so):
- Units have **dependencies/ordering** (pipelines, DAGs, "can't start X until Y",
  set-level reasoning where one item reframes others). This model assumes
  independence — do not pretend otherwise.
- **Correctness/quality** is the hard part. This models "done", never "done
  right".
- Dispatch is **centralized and instant** (a normal job queue). Then there is no
  search cost — use plain **queueing theory** (Little's Law, M/G/k) instead.
- The cost is **compute/IO**, not "agent attention".

If it's a poor fit, tell the user plainly, name the better tool (queueing theory,
a DAG/workflow scheduler, an evals/quality framework), and stop. A correct "this
doesn't fit" is a successful outcome.

## Step 2 — Gather inputs

Ask for (or infer, stating assumptions):
- **Batch size** N (number of work units).
- **Difficulty mix**: fraction that is "hard", and how much harder (≈ how many
  units of effort a hard item is vs. an easy one).
- **Agent tiers**: for each tier, the count, the throughput (items or bites per
  second/minute/hour), and **whether that tier can handle hard items**.
- **Deadline / SLA**, if any.
- **Concurrent workloads**: is this one batch, or several projects sharing the
  pool at once? (Maps to `--workloads K`; the benchmark then reports a fairness
  gap between the first and last workload to finish.)
- **Is discovery costly?** (decentralized assignment, agents exploring, humans
  choosing what to pick up) - confirms the model applies.

## Step 3 — Capacity estimate and bottleneck (fast, no simulation)

Run the analytic estimator from the project root:

```bash
.venv/bin/python fish_food.py --theory --hard-fraction <F> --observed <secs>
```

Map the real workload onto these parameters (edit the `Config` dataclass /
`FishType` rows in `fish_food.py`, or pass flags, to mirror the real tiers):

| Real-world concept | Fish Food parameter |
|---|---|
| A work unit | a pellet (`n_pellets`) |
| Hard/complex items | `hard_fraction`, `hard_bites` |
| Item needs an expert | `hard_min_mouth` (only big `mouth` qualifies) |
| Worker tier size | `FishType.count` |
| Worker throughput | `FishType.cooldown` (rate = count / cooldown) |
| Worker capability | `FishType.mouth` (>= `hard_min_mouth` => can do hard) |
| Cheap worker daily cap | `FishType.max_eat` |
| Routing rule | `--policy greedy|specialist` |

If you reason by hand, use:
- per-tier rate `r_i = count_i / cooldown_i`
- `R_total = sum r_i`; `R_capable = sum r_i over tiers that can do hard`
- total work (in bites) `W = easy*1 + hard*hard_bites`
- **optimistic floor** `T_floor = max(W / R_total, hard_work / R_capable)`
- the larger term names the **bottleneck** (whole-pool capacity vs. the scarce
  capable tier).

Report the floor, the bottleneck tier, and — if an observed mean is supplied —
the **utilization** (`floor / observed`) and the **discovery/search overhead**
(`observed / floor`). Empirically this overhead is large (often ~8x) because
real systems spend most time finding/coordinating, not working.

## Step 4 — Predictability (calibrated simulation, optional but powerful)

If the user wants the completion-time *distribution* (not just a floor), run a
benchmark with parameters calibrated to the real workload:

```bash
.venv/bin/python fish_food.py --runs 20 --pellets <N> --hard-fraction <F> --plot dist.png
```

Interpret the **CV%** (coefficient of variation): lower = more constant-time =
more reliable for an SLA. Rough reading: <8% very predictable, 8–15% fairly,
15–30% moderate, >30% unreliable.

Toggle behaviors to test levers:
- `--policy specialist` — reserve capable agents for hard items.
- `--legacy-search` — turn off recruitment + area-restricted search (baseline).
- `--workloads K` — several projects sharing one pool; watch the **fairness
  gap** (pooling keeps aggregate throughput but can starve individual
  workloads). Pooling is free on throughput, not on per-project predictability.
- `--staggered --arrival-interval S` — workloads arrive over time (a queue),
  not all at once.
- Note: the best search strategy depends on the work's spatial structure.
  Recruitment/ARS help with a clumped single-workload tail but HURT when work
  is pre-distributed across many sites (try `--legacy-search` for multi-site).
- `--recruit-radius / --recruit-recent / --ars-*` — search-tuning knobs to
  sweep for lower CV (defaults are already near-optimal; expect a
  speed-vs-predictability tradeoff).

Note: large batches are compute-heavy. Use a smaller `--pellets` for quick
iteration and tell the user the result is relative, then scale up once the
parameters are right.

## Step 5 — Recommendations (apply the model's lessons)

Translate findings into advice, grounded in these established results:
- **Speed is capacity-bound.** If the bottleneck is the scarce capable tier,
  adding *cheap* workers does almost nothing — add/strengthen capable agents or
  reduce per-item effort.
- **Predictability is a separate dial** bought by better discovery: recruitment
  (idle agents pulled toward active work) and area-restricted search roughly
  *halve* completion-time variance at little cost to the mean.
- **Task-selection routing alone (specialist) does NOT fix a search-limited
  system** — reducing discovery cost does. Diagnose which regime you're in via
  the utilization number from Step 3.
- Match agent capability to item difficulty; keep the scarce tier fed.

## Step 6 — Report

Produce a concise report with: fit verdict; the bottleneck and capacity floor;
the predicted completion window + CV (predictable or not); the top 1–3 levers
ranked by expected impact; and explicit assumptions + what to calibrate with real
measurements. Always state the two hard limits: this ignores **dependencies**
and **quality/correctness**.

## Caveats to repeat to the user
- Outputs are **structural** unless calibrated with the user's measured rates and
  difficulty mix.
- It supports **multiple concurrent workloads** sharing one pool (`--workloads`)
  with optional **staggered arrivals** (`--staggered`), but does not yet model
  per-workload priorities/SLAs or preemption.
- It is **not** a model of set-level/dependency reasoning (e.g. tracing a chain
  of title across documents) and **not** a substitute for quality evaluation.
