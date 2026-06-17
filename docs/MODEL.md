# Fish Food — the model and its equations

Fish Food is an **agent-based model (ABM)**: completion time *emerges* from many
interacting agents, so there is **no single closed-form equation** `T = f(...)`
for it. It is, however, precisely described at three levels:

1. a closed-form **capacity floor** (a lower bound on time),
2. an **empirical relation** linking that floor to observed time and to
   predictability, and
3. the **microdynamics** — the per-step update rules the simulator integrates,
   which fully *define* the algorithm.

Units are meters and seconds. All symbols are collected at the bottom.

---

## Level 1 — Capacity floor (closed form)

This is what `fish_food.py --theory` computes.

Work content (measured in "bites"):

```
H  = round(f_h · N)          # number of hard units
E  = N − H                   # number of easy units
W  = E·1 + H·b_h             # total work, in bites
```

Per-tier sustained bite rate, and the two pool rates:

```
r_i      = n_i / c_i                      # tier i: count / cooldown
R_total  = Σ_i r_i                         # all consumers
R_hard   = Σ_{i : mouth_i ≥ m*} r_i        # only consumers able to do hard work
```

Optimistic completion-time **floor** (assumes perfect utilization and zero
travel/search):

```
T_floor = max( W / R_total ,  (H·b_h) / R_hard )
```

- The first term is the whole-pool throughput limit.
- The second is the **bottleneck** when hard units can only be done by the scarce
  capable tier. Whichever term is larger names the binding constraint.

If `R_hard = 0` while `H > 0`, the batch can never finish (the simulator raises
an error for this case).

---

## Level 2 — Observed time and predictability (empirical)

Real runs sit **above** the floor by a travel/search overhead factor `η`:

```
T_obs ≈ η · T_floor          # measured η ≈ 8 in our experiments
U      = T_floor / T_obs     # utilization = 1 / η
```

The project's headline property is *predictability*, the coefficient of
variation of completion time across random seeds:

```
μ_T = mean(T over seeds)
σ_T = stdev(T over seeds)
CV  = σ_T / μ_T              # small CV  ⇒  "always gone in about the same time"
```

`η` and `CV` have no closed form — they depend on density, sense range, spatial
layout, and search behavior. That dependence is exactly what the simulator
exists to measure. Empirically:

- **Speed is capacity-bound.** When the bottleneck is `R_hard`, adding cheap
  workers barely changes `T_obs`; only more/faster capable workers (raising
  `R_hard`) or less work per item (lower `b_h`) move it.
- **Predictability is a separate dial.** Recruitment + area-restricted search
  roughly halve `CV` on a clumped tail — but *raise* it when work is
  pre-distributed across many sites (they over-concentrate the pool).

---

## Level 3 — Microdynamics (the algorithm, per timestep Δt)

These are the update rules integrated each step. `xⱼ, vⱼ` are a unit's position
and velocity; `xᵢ, vᵢ` a consumer's. `dᵢⱼ = |xⱼ − xᵢ|`.

### Work units (advection)

```
aⱼ =  Σ_{i awake, dᵢⱼ < ρᵢ}  Pᵢ · (1 − dᵢⱼ/ρᵢ) · (xⱼ − xᵢ)/dᵢⱼ      # consumer wakes
    + P_pump · (1 − d/ρ_pump) · (xⱼ − p)/d        (if d < ρ_pump)    # pump
    + 𝒩(0, τ²)                                                       # turbulence

vⱼ ← (vⱼ + aⱼ) · (1 − k_drag · Δt)        # impulse + water drag
xⱼ ← xⱼ + vⱼ · Δt                          # integrate
# at a wall or lily pad: clamp xⱼ inside, mark stuck, vⱼ ← s_damp · vⱼ
```

Units also have a **drop time**: they only participate once `t ≥ drop_tⱼ`
(progressive fill; with staggered arrivals, `drop_tⱼ` includes
`workloadⱼ · arrival_interval`).

### Consumers (steering)

```
# "workable" unit = within sense range sᵢ AND mouthᵢ ≥ mⱼ*
desiredᵢ = (x* − xᵢ)/|x* − xᵢ| · speedᵢ      if a workable unit x* is in range
         = ŵᵢ · speedᵢ                        otherwise (wander heading ŵᵢ)
# wander heading does a random walk; turn strength is ARS-modulated by time
#   since last bite. Idle searchers may be recruited toward a nearby feeder.
desiredᵢ = 0                                  if asleep (not yet awake)

vᵢ ← vᵢ + α · (desiredᵢ − vᵢ),   α = 0.25     # smooth steering
|vᵢ| ← min(|vᵢ|, speedᵢ)                       # speed clamp
xᵢ ← xᵢ + vᵢ · Δt                              # integrate (+ wall bounce)
```

A consumer **wakes** when `t ≥ wake_tᵢ` or a unit drifts within `alert_radius`.

### Eating (one bite per ready consumer per step)

```
cooldownᵢ ← max(0, cooldownᵢ − Δt)
ready  = (cooldownᵢ ≤ 0) ∧ (eatenᵢ < maxeatᵢ) ∧ awakeᵢ
# pick the nearest unit j with dᵢⱼ < mouthᵢ and mouthᵢ ≥ mⱼ* (specialist policy
# adds a distance penalty to easy units for capable consumers)
if such a unit exists:
    bitesⱼ   ← bitesⱼ − 1
    cooldownᵢ ← cᵢ
    if bitesⱼ ≤ 0:  unit j is finished
```

### Completion

```
T = min{ t : bitesⱼ ≤ 0  for all units j }
```

For K concurrent workloads, the per-workload finish time is
`Tᵂ = min{ t : all units of workload w finished }`, and the **fairness gap** is
`max_w Tᵂ − min_w Tᵂ`.

---

## Symbols

| symbol | meaning |
|---|---|
| `N`, `E`, `H` | total / easy / hard unit counts |
| `f_h`, `b_h` | hard fraction (`hard_fraction`), bites per hard unit (`hard_bites`) |
| `m*` | min mouth to work a hard unit (`hard_min_mouth`); `mⱼ*` is unit j's requirement |
| `n_i`, `c_i` | tier i consumer count, bite cooldown |
| `r_i`, `R_total`, `R_hard` | tier rate, all-pool rate, hard-capable rate (bites/s) |
| `W`, `T_floor` | total work (bites), capacity-floor time |
| `η`, `U`, `CV` | search overhead, utilization, completion-time coefficient of variation |
| `speedᵢ`, `sᵢ`, `mouthᵢ` | consumer max speed, sense radius, mouth (eat) radius |
| `Pᵢ`, `ρᵢ` | consumer wake push strength, wake radius |
| `P_pump`, `ρ_pump`, `p` | pump strength, radius, position |
| `k_drag`, `τ`, `s_damp` | water drag, turbulence std, stuck-velocity damping |
| `Δt` | timestep (`dt`) |

All of these are fields in the `Config` / `FishType` dataclasses in
`fish_food.py`.

---

## What the model does *not* capture

- **Dependencies / ordering** between units (it assumes independence) — so it
  does not model set-level reasoning such as tracing a chain of title.
- **Correctness / quality** — it models "done", never "done right".
- **Per-workload priorities / SLAs or preemption.**

Use the [`workload-capacity-planner`](../.agents/skills/workload-capacity-planner/SKILL.md)
skill's fit-triage before applying this to a real workload.
