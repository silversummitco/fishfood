#!/usr/bin/env python3
# Fish Food - https://github.com/silversummitco/fishfood
# Copyright (c) 2026 SilverSummitCo LLC.  Licensed under the PolyForm
# Noncommercial License 1.0.0 - see the LICENSE file. Free for non-commercial
# use; commercial/profit use requires a commercial license - see COMMERCIAL.md.
# Required Notice: Copyright (c) 2026 SilverSummitCo LLC
# (https://github.com/silversummitco/fishfood)
"""Fish Food - an agent-based pond-feeding simulator.

The idea: a fixed handful of fish-food
pellets is dropped in the center of a pond. A heterogeneous school of fish
(big fast Koi, juveniles, goldfish, and hundreds of tiny minnows) swarm the
clump. Their motion both *eats* the pellets and *stirs the water*, advecting
the clump outward toward the walls and lily pads where leftovers stick.

The claim worth seeing: the batch is always consumed in roughly the same
(bounded, near-constant) time regardless of the random initial scatter - the
way a handful of pellets is always gone in about three minutes.

Run modes
---------
Visual (default), opens a pygame window you can watch:

    python fish_food.py

Headless benchmark, runs many sims with no window and reports the spread of
completion times to test the constant-time claim:

    python fish_food.py --runs 20

pygame is imported lazily inside the visual code path, so the benchmark runs
fine with no display or without pygame installed.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from typing import final

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Units are meters and seconds. Defaults are first-guess values; everything
# here is meant to be tuned (see the README's "Tuning" section).


@dataclass
class FishType:
    """Per-species parameters. One row per individual is built from these."""

    name: str
    count: int
    speed: float  # max swim speed (m/s)
    mouth: float  # eat radius (m): pellet within this is eaten
    sense: float  # detection radius (m): chase nearest pellet inside
    push_radius: float  # wake radius (m): pellets inside get shoved
    push: float  # wake strength (acceleration scale)
    cooldown: float  # seconds between bites
    max_eat: float  # lifetime pellet cap (inf = unlimited)
    draw_size: float  # visual radius (m), purely cosmetic
    color: tuple[int, int, int]


@dataclass
class Config:
    # Pond
    pond_w: float = 6.0
    pond_h: float = 4.0

    # Pellets (the batch / data)
    n_pellets: int = 3000
    clump_radius: float = 0.20  # initial "basketball" clump radius (m)

    # Concurrent workloads. 1 = the classic single pond. With >1, the SAME
    # n_pellets of total work is split into this many separate clumps dropped at
    # different spots, all served by the one shared agent pool - modeling a
    # fleet handling several projects at once. Lets you study fairness (do all
    # workloads finish together, or does the pool swarm one and starve others?).
    n_workloads: int = 1

    # Staggered arrivals (off by default, so the classic pond is unchanged).
    # When on with >1 workload, each workload's clump drops a bit later than the
    # last - modeling a real queue of jobs arriving over time instead of all at
    # once. The shared pool clears each as it lands, idling between arrivals.
    staggered_arrivals: bool = False
    arrival_interval: float = 8.0  # seconds between successive workload arrivals

    # Workload heterogeneity. Default (0.0) is a uniform batch where every unit
    # is one easy bite - the classic pond. Turn this up to model data of mixed
    # difficulty (e.g. a simple deed vs. a tangled chain of title): a fraction
    # of units become "hard", needing several bites AND a large enough mouth,
    # so only the big consumers can finish them.
    hard_fraction: float = 0.0  # fraction of units that are "hard"
    hard_bites: int = 4  # bites to finish a hard unit (easy = 1 bite)
    hard_min_mouth: float = 0.08  # min consumer mouth size to work a hard unit

    # Scheduling / routing policy - who works what.
    #   "greedy"     : every consumer chases the nearest unit it can work.
    #   "specialist" : big consumers (those able to work hard units) prefer
    #                  hard units, reserving the scarce heavyweight workers for
    #                  the bottleneck work instead of grabbing easy units the
    #                  swarm could handle.
    policy: str = "greedy"
    specialist_bias: float = 5.0  # how strongly big consumers avoid easy units (m)

    # Search behavior - the lever that attacks the travel/search overhead.
    # Recruitment: idle searchers drift toward fish that are actively feeding
    # (like a school converging on a feeding frenzy), concentrating effort
    # where work actually is instead of wandering blindly.
    recruit: bool = True
    recruit_radius: float = 1.3  # idle fish steer toward a feeder within this (m)
    recruit_recent: float = 0.6  # a fish counts as "feeding" if it ate this recently
    # Area-restricted search: turn sharply right after eating (stay in the patch),
    # straighten out into ballistic dispersal when food hasn't been found lately.
    ars: bool = True
    ars_full_turn: float = 1.2  # wander turn strength just after a find (local)
    ars_lost_turn: float = 0.12  # wander turn strength when long unfed (ballistic)
    ars_memory: float = 6.0  # seconds to ramp from local search to dispersal

    # Pump: a fixed point pushing water (and pellets) gently outward
    pump_pos: tuple[float, float] = (3.0, 2.0)  # defaults to center
    pump_radius: float = 0.6
    pump_push: float = 0.05

    # Water / surface dynamics
    drag: float = 2.2  # velocity damping per second
    turbulence: float = 0.015  # random jitter added to pellet velocity
    wall_margin: float = 0.06  # pellet within this of a wall "sticks"
    stick_damp: float = 0.06  # velocity multiplier for stuck pellets

    # Startup choreography (matches the real pond: food drops in, fish doze)
    settle_seconds: float = 3.0  # fish doze while the handful is dropped
    drop_window: float = 2.5  # pellets appear over this long, filling the clump
    wake_spread: float = 4.0  # fish wake staggered across this window
    alert_radius: float = 0.9  # a fish also wakes if food drifts this near

    # Lily pads: discs that trap/damp pellets (x, y, radius)
    lily_pads: tuple[tuple[float, float, float], ...] = (
        (1.2, 1.0, 0.45),
        (4.7, 3.0, 0.55),
        (4.9, 0.9, 0.35),
    )

    # Simulation
    dt: float = 0.05
    max_seconds: float = 1200.0  # safety cap so a run can't hang forever

    # Fish school (heterogeneous - this is the interesting part)
    fish_types: tuple[FishType, ...] = field(
        default_factory=lambda: (
            FishType(
                "Adult Koi",
                2,
                0.55,
                0.14,
                2.2,
                0.40,
                0.90,
                0.10,
                math.inf,
                0.16,
                (235, 120, 40),
            ),
            FishType(
                "Juvenile Koi",
                4,
                0.45,
                0.09,
                1.6,
                0.28,
                0.50,
                0.20,
                math.inf,
                0.10,
                (240, 180, 70),
            ),
            FishType(
                "Goldfish",
                3,
                0.34,
                0.06,
                1.2,
                0.18,
                0.30,
                0.40,
                math.inf,
                0.07,
                (245, 150, 60),
            ),
            FishType(
                "Minnow",
                400,
                0.25,
                0.03,
                0.6,
                0.07,
                0.05,
                2.50,
                6.0,
                0.03,
                (180, 200, 220),
            ),
        )
    )


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@final
class Simulation:
    """Vectorized pellet + fish state. One instance == one pond run."""

    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.steps = 0

        self._init_pellets()
        self._init_fish()

    # -- setup ------------------------------------------------------------
    def _workload_centers(self) -> np.ndarray:
        """Drop-in location for each workload's clump."""
        cfg = self.cfg
        cx, cy = cfg.pond_w / 2.0, cfg.pond_h / 2.0
        k = max(1, cfg.n_workloads)
        if k == 1:
            return np.array([[cx, cy]])
        # Spread the clumps evenly on a ring so the shared pool has to cover
        # genuinely separate regions.
        radius = 0.32 * min(cfg.pond_w, cfg.pond_h)
        ang = np.linspace(0.0, 2.0 * np.pi, k, endpoint=False)
        return np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)

    def _init_pellets(self) -> None:
        cfg = self.cfg
        n = cfg.n_pellets
        k = max(1, cfg.n_workloads)

        # Split the total work across k workloads (as evenly as possible) and
        # give each its own clump location. Each pellet remembers its workload.
        self.n_wk = k
        base, rem = divmod(n, k)
        counts = np.full(k, base, dtype=int)
        counts[:rem] += 1
        self.p_workload = np.repeat(np.arange(k), counts)
        centers = self._workload_centers()[self.p_workload]  # (n, 2)

        # Tight clump per workload (uniform over a small disc) - each clump
        # fills its whole basketball-sized circle, not just its rim.
        r = cfg.clump_radius * np.sqrt(self.rng.random(n))
        a = self.rng.random(n) * 2.0 * np.pi
        self.p_pos = centers + np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
        self.p_vel = np.zeros((n, 2))
        self.p_alive = np.ones(n, dtype=bool)
        self.p_stuck = np.zeros(n, dtype=bool)
        # Pellets appear progressively, so the clump visibly fills up over the
        # first couple of seconds rather than popping in all at once.
        self.p_drop_t = self.rng.random(n) * cfg.drop_window
        # Optional staggered arrivals: each workload lands later than the last,
        # modeling a queue of jobs arriving over time.
        if cfg.staggered_arrivals and k > 1:
            self.p_drop_t = self.p_drop_t + self.p_workload * cfg.arrival_interval

        # Work content: bites remaining to finish a unit, and the minimum
        # consumer mouth size able to work it. Easy units = 1 bite, no minimum.
        self.p_bites = np.ones(n)
        self.p_minmouth = np.zeros(n)
        if cfg.hard_fraction > 0.0:
            n_hard = int(round(n * cfg.hard_fraction))
            if n_hard > 0:
                hard_idx = self.rng.choice(n, size=n_hard, replace=False)
                self.p_bites[hard_idx] = float(cfg.hard_bites)
                self.p_minmouth[hard_idx] = cfg.hard_min_mouth
                # Feasibility guard: someone must be able to finish hard units,
                # or the batch can never complete.
                biggest_mouth = max(ft.mouth for ft in cfg.fish_types)
                if biggest_mouth < cfg.hard_min_mouth:
                    raise ValueError(
                        f"No consumer can work hard units: largest mouth "
                        f"{biggest_mouth} < hard_min_mouth {cfg.hard_min_mouth}."
                    )

        # When each workload is fully consumed, we stamp the time here.
        self.workload_done_t = np.full(self.n_wk, np.nan)

    def _init_fish(self) -> None:
        cfg = self.cfg
        rows: list[FishType] = []
        for ft in cfg.fish_types:
            rows.extend([ft] * ft.count)
        m = len(rows)

        self.f_speed = np.array([f.speed for f in rows])
        self.f_mouth = np.array([f.mouth for f in rows])
        self.f_sense = np.array([f.sense for f in rows])
        self.f_pushR = np.array([f.push_radius for f in rows])
        self.f_push = np.array([f.push for f in rows])
        self.f_cooldown = np.array([f.cooldown for f in rows])
        self.f_maxeat = np.array([f.max_eat for f in rows])
        self.f_drawsize = np.array([f.draw_size for f in rows])
        self.f_color = np.array([f.color for f in rows], dtype=np.uint8)
        # "big" = able to work hard units; used by the specialist policy.
        self.f_big = self.f_mouth >= cfg.hard_min_mouth
        # index of the FishType each fish belongs to (for HUD / coloring)
        self.f_type = np.concatenate(
            [np.full(ft.count, i) for i, ft in enumerate(cfg.fish_types)]
        )

        # Start fish spread randomly around the pond (they "arrive").
        margin = 0.2
        self.f_pos = np.stack(
            [
                self.rng.uniform(margin, cfg.pond_w - margin, m),
                self.rng.uniform(margin, cfg.pond_h - margin, m),
            ],
            axis=1,
        )
        ang = self.rng.random(m) * 2.0 * np.pi
        # Fish start nearly motionless - dozing until the food wakes them.
        self.f_vel = self.rng.normal(0.0, 0.01, (m, 2))

        self.f_cool_timer = np.zeros(m)  # time until this fish may bite
        self.f_eaten = np.zeros(m, dtype=int)  # lifetime pellets eaten
        self.f_wander = ang.copy()  # current wander heading
        # Time since this fish last took a bite (drives area-restricted search
        # and recruitment). Starts "long ago" so nobody is locally searching yet.
        self.f_since_ate = np.full(m, 1e3)
        # Each fish wakes after its own staggered delay (or sooner if food
        # drifts within alert_radius - see step()).
        self.f_wake_t = cfg.settle_seconds + self.rng.random(m) * cfg.wake_spread
        self.f_awake = np.zeros(m, dtype=bool)

    # -- per-step physics -------------------------------------------------
    @property
    def alive_count(self) -> int:
        return int(self.p_alive.sum())

    @property
    def done(self) -> bool:
        return self.alive_count == 0

    @property
    def present_mask(self) -> np.ndarray:
        """Pellets that have dropped in and are not yet eaten."""
        return self.p_alive & (self.t >= self.p_drop_t)

    def step(self) -> None:
        cfg = self.cfg
        dt = cfg.dt
        present = self.present_mask

        if present.any():
            ai = np.where(present)[0]  # indices of dropped, uneaten pellets
            P = self.p_pos[ai]  # (Na, 2)
            F = self.f_pos  # (M, 2)

            # Pairwise fish->pellet offsets/distances. M is a few hundred, Na
            # up to a couple thousand, so the dense matrix stays manageable.
            diff = P[None, :, :] - F[:, None, :]  # (M, Na, 2) pellet - fish
            dist2 = np.einsum("mnk,mnk->mn", diff, diff)  # (M, Na)
            dist = np.sqrt(dist2)

            # Wake fish: on their own timer, or the moment food drifts close.
            nearest_d = dist.min(axis=1)
            self.f_awake |= (self.t >= self.f_wake_t) | (nearest_d < cfg.alert_radius)

            self._fish_behavior(diff, dist, ai, dt)
            self._advect_pellets(diff, dist, ai, dt)
            self._eat(dist, ai)
        else:
            self._idle(dt)

        # Stamp the finish time of any workload that just emptied.
        if self.n_wk > 1:
            remaining = np.bincount(self.p_workload[self.p_alive], minlength=self.n_wk)
            newly_done = (remaining == 0) & np.isnan(self.workload_done_t)
            if newly_done.any():
                self.workload_done_t[newly_done] = self.t

        self.t += dt
        self.steps += 1

    def _idle(self, dt) -> None:
        """No food in the water yet (or all eaten): fish just doze in place."""
        cfg = self.cfg
        self.f_vel *= 0.55
        self.f_vel += self.rng.normal(0.0, 0.004, self.f_vel.shape)
        self.f_pos += self.f_vel * dt
        np.clip(self.f_pos[:, 0], 0.05, cfg.pond_w - 0.05, out=self.f_pos[:, 0])
        np.clip(self.f_pos[:, 1], 0.05, cfg.pond_h - 0.05, out=self.f_pos[:, 1])

    def _fish_behavior(self, diff, dist, ai, dt) -> None:
        """Steer each fish toward its nearest sensed pellet, else wander."""
        cfg = self.cfg
        M = self.f_pos.shape[0]

        awake = self.f_awake
        self.f_since_ate += dt  # ages every fish's "time since last bite"

        # nearest sensed pellet this fish can actually work (within sense range
        # AND big enough mouth for the unit's difficulty), per fish
        can_work = self.f_mouth[:, None] >= self.p_minmouth[None, ai]  # (M, Na)
        within = (dist < self.f_sense[:, None]) & can_work
        masked = np.where(within, dist, np.inf)  # (M, Na)
        # Specialist policy: big consumers add a distance penalty to easy units
        # so they prefer (travel toward) hard units when any are in range.
        if cfg.policy == "specialist":
            easy_unit = (self.p_minmouth[ai] == 0.0)[None, :]  # (1, Na)
            penalize = self.f_big[:, None] & easy_unit & within
            masked = np.where(penalize, masked + cfg.specialist_bias, masked)
        nearest = np.argmin(masked, axis=1)  # (M,)
        has_target = np.isfinite(masked[np.arange(M), nearest])

        # direction toward target (diff is pellet - fish, i.e. toward pellet)
        tgt_off = diff[np.arange(M), nearest]  # (M, 2)
        tgt_dist = dist[np.arange(M), nearest][:, None]
        tgt_dir = np.where(tgt_dist > 1e-9, tgt_off / np.maximum(tgt_dist, 1e-9), 0.0)

        # wander: heading does a random walk. Under area-restricted search the
        # turn strength depends on how long since this fish last ate: turn hard
        # right after a find (stay in the patch), straighten out into ballistic
        # dispersal when food has been scarce (cover new ground).
        if cfg.ars:
            frac = np.clip(self.f_since_ate / cfg.ars_memory, 0.0, 1.0)
            turn_mag = cfg.ars_full_turn * (1.0 - frac) + cfg.ars_lost_turn * frac
        else:
            turn_mag = 0.6
        self.f_wander += self.rng.normal(0.0, 1.0, M) * turn_mag * 0.15
        wander_dir = np.stack([np.cos(self.f_wander), np.sin(self.f_wander)], axis=1)

        # recruitment: idle searchers drift toward the nearest actively-feeding
        # fish, concentrating effort where work is instead of wandering blindly.
        if cfg.recruit:
            feeding = self.f_since_ate < cfg.recruit_recent
            searching = awake & (~has_target) & (~feeding)
            if feeding.any() and searching.any():
                f_idx = np.where(feeding)[0]
                s_idx = np.where(searching)[0]
                dvec = self.f_pos[f_idx][None, :, :] - self.f_pos[s_idx][:, None, :]
                dd = np.sqrt(np.einsum("skc,skc->sk", dvec, dvec))
                nn = np.argmin(dd, axis=1)
                rows = np.arange(s_idx.size)
                nd = dd[rows, nn]
                close = nd < cfg.recruit_radius
                if close.any():
                    toward = dvec[rows, nn] / np.maximum(nd[:, None], 1e-9)
                    wander_dir[s_idx[close]] = toward[close]

        desired_dir = np.where(has_target[:, None], tgt_dir, wander_dir)
        desired_vel = desired_dir * self.f_speed[:, None]
        desired_vel[~awake] = 0.0  # dozing fish don't try to swim anywhere

        # smooth steering toward desired velocity
        turn = 0.25
        self.f_vel += (desired_vel - self.f_vel) * turn

        # dozing fish settle to near-rest with only a faint idle drift
        asleep = ~awake
        if asleep.any():
            self.f_vel[asleep] *= 0.55
            self.f_vel[asleep] += self.rng.normal(0.0, 0.004, (int(asleep.sum()), 2))

        # clamp to max speed
        sp = np.linalg.norm(self.f_vel, axis=1, keepdims=True)
        too_fast = (sp > self.f_speed[:, None]).ravel()
        self.f_vel[too_fast] *= (self.f_speed[too_fast] / sp.ravel()[too_fast])[:, None]

        # integrate + bounce off walls
        self.f_pos += self.f_vel * dt
        for axis, hi in ((0, cfg.pond_w), (1, cfg.pond_h)):
            lo_hit = self.f_pos[:, axis] < 0.05
            hi_hit = self.f_pos[:, axis] > hi - 0.05
            self.f_pos[lo_hit, axis] = 0.05
            self.f_pos[hi_hit, axis] = hi - 0.05
            self.f_vel[lo_hit | hi_hit, axis] *= -1.0
            # nudge wander heading away from the wall it just hit
            self.f_wander[lo_hit | hi_hit] = (
                self.rng.random((lo_hit | hi_hit).sum()) * 2 * np.pi
            )

    def _advect_pellets(self, diff, dist, ai, dt) -> None:
        """Move pellets: fish wakes + pump + turbulence, with drag & sticking."""
        cfg = self.cfg
        Na = ai.size

        # --- fish wakes: each fish shoves nearby pellets radially outward ---
        inwake = dist < self.f_pushR[:, None]  # (M, Na)
        # falloff 1 at the fish, 0 at the wake edge. Only awake (thrashing)
        # fish stir the water - dozing fish leave the clump undisturbed.
        falloff = np.clip(1.0 - dist / self.f_pushR[:, None], 0.0, 1.0)
        awake_f = self.f_awake[:, None].astype(float)
        scale = np.where(inwake, falloff * self.f_push[:, None] * awake_f, 0.0)
        invd = 1.0 / np.maximum(dist, 1e-6)
        # diff is (pellet - fish): pointing away from fish == outward push
        push_vec = diff * (scale * invd)[:, :, None]  # (M, Na, 2)
        accel = push_vec.sum(axis=0)  # (Na, 2)

        # --- pump: constant gentle outward push near the pump point ---
        pump = np.asarray(cfg.pump_pos)
        d = self.p_pos[ai] - pump  # (Na, 2)
        pd = np.linalg.norm(d, axis=1)
        pmask = pd < cfg.pump_radius
        pdir = np.where(pd[:, None] > 1e-6, d / np.maximum(pd[:, None], 1e-6), 0.0)
        pump_falloff = np.clip(1.0 - pd / cfg.pump_radius, 0.0, 1.0)
        accel += pdir * (pmask * pump_falloff * cfg.pump_push)[:, None]

        # --- turbulence: small random surface jitter ---
        accel += self.rng.normal(0.0, cfg.turbulence, (Na, 2))

        v = self.p_vel[ai]
        v += accel
        v *= max(0.0, 1.0 - cfg.drag * dt)  # water drag
        pos = self.p_pos[ai] + v * dt

        # --- walls: clamp inside pond and mark as stuck ---
        stuck = self.p_stuck[ai]
        for axis, hi in ((0, cfg.pond_w), (1, cfg.pond_h)):
            lo = pos[:, axis] < cfg.wall_margin
            hh = pos[:, axis] > hi - cfg.wall_margin
            pos[lo, axis] = np.clip(pos[lo, axis], 0.0, hi)
            pos[hh, axis] = np.clip(pos[hh, axis], 0.0, hi)
            stuck |= lo | hh

        # --- lily pads: trap pellets that drift under a pad ---
        for lx, ly, lr in cfg.lily_pads:
            on_pad = (pos[:, 0] - lx) ** 2 + (pos[:, 1] - ly) ** 2 < lr * lr
            stuck |= on_pad

        # stuck pellets have their velocity strongly damped
        v[stuck] *= cfg.stick_damp

        self.p_vel[ai] = v
        self.p_pos[ai] = pos
        self.p_stuck[ai] = stuck

    def _eat(self, dist, ai) -> None:
        """Each ready fish eats its nearest pellet within mouth range."""
        dt = self.cfg.dt
        self.f_cool_timer = np.maximum(0.0, self.f_cool_timer - dt)

        ready = (
            (self.f_cool_timer <= 0.0) & (self.f_eaten < self.f_maxeat) & self.f_awake
        )
        if not ready.any():
            return

        # eligible[m, j] : fish m may bite present-pellet j -- within its mouth
        # reach AND mouth large enough for that unit's difficulty.
        pminmouth = self.p_minmouth[ai]  # (Na,)
        eligible = (
            (dist < self.f_mouth[:, None])
            & ready[:, None]
            & (self.f_mouth[:, None] >= pminmouth[None, :])
        )
        if not eligible.any():
            return

        # Resolve greedily over the (few) ready+eligible fish. Closest first
        # so a unit goes to the nearest qualified mouth. Under the specialist
        # policy, big consumers prefer hard units over easy ones in reach.
        masked = np.where(eligible, dist, np.inf)
        if self.cfg.policy == "specialist":
            easy_unit = (self.p_minmouth[ai] == 0.0)[None, :]
            penalize = self.f_big[:, None] & easy_unit & eligible
            masked = np.where(penalize, masked + self.cfg.specialist_bias, masked)
        cand_j = np.argmin(masked, axis=1)  # nearest workable unit per fish
        cand_d = masked[np.arange(masked.shape[0]), cand_j]
        biters = np.where(np.isfinite(cand_d))[0]
        # nearest mouths get first pick (one bite per unit per step)
        biters = biters[np.argsort(cand_d[biters])]

        bitten_local = set()
        for m in biters:
            j = int(cand_j[m])
            if j in bitten_local:
                continue
            bitten_local.add(j)
            pellet = int(ai[j])
            # Take one bite. Hard units need several bites (possibly from
            # several fish over several cooldowns) before they're finished.
            self.p_bites[pellet] -= 1.0
            self.f_eaten[m] += 1  # counts work done (bites), feeds the eat cap
            self.f_cool_timer[m] = self.f_cooldown[m]
            self.f_since_ate[m] = 0.0  # just fed: search locally / recruit others
            if self.p_bites[pellet] <= 0.0:
                self.p_alive[pellet] = False


# --------------------------------------------------------------------------
# Headless benchmark
# --------------------------------------------------------------------------


def run_headless(cfg: Config, seed: int | None) -> dict:
    """Run one sim to completion with no window. Returns summary stats."""
    sim = Simulation(cfg, seed=seed)
    max_steps = int(cfg.max_seconds / cfg.dt)
    while not sim.done and sim.steps < max_steps:
        sim.step()
    wk = sim.workload_done_t
    finished = wk[~np.isnan(wk)]
    # Fairness gap = spread between the first and last workload to finish.
    spread = float(finished.max() - finished.min()) if finished.size > 1 else 0.0
    return {
        "seed": seed,
        "time": sim.t,
        "steps": sim.steps,
        "remaining": sim.alive_count,
        "completed": sim.done,
        "workload_times": wk.tolist(),
        "workload_spread": spread,
    }


def _ascii_histogram(values: list[float], bins: int = 12, width: int = 40) -> list[str]:
    """A tiny text histogram so the distribution is visible without a GUI."""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [f"    all {len(values)} run(s) finished at ~{lo:.1f}s"]
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / (hi - lo) * bins))
        counts[idx] += 1
    cmax = max(counts) or 1
    lines = []
    for k in range(bins):
        e0 = lo + (hi - lo) * k / bins
        e1 = lo + (hi - lo) * (k + 1) / bins
        bar = "#" * int(round(counts[k] / cmax * width))
        lines.append(f"  {e0:6.1f}-{e1:6.1f}s | {bar:<{width}} {counts[k]}")
    return lines


def _save_histogram_png(values: list[float], path: str, bins: int = 12) -> None:
    """Save a simple completion-time histogram PNG (uses Pillow if available)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        print(f"  (Pillow not installed; skipped writing {path})")
        return
    W, H, m = 720, 360, 50
    lo, hi = min(values), max(values)
    span = max(hi - lo, 1e-9)
    counts = [0] * bins
    for v in values:
        counts[min(bins - 1, int((v - lo) / span * bins))] += 1
    cmax = max(counts) or 1
    img = Image.new("RGB", (W, H), (18, 26, 34))
    d = ImageDraw.Draw(img)
    d.rectangle([m, 20, W - 20, H - m], outline=(70, 90, 110))
    bw = (W - 20 - m) / bins
    for k in range(bins):
        bh = (H - m - 20) * counts[k] / cmax
        x0 = m + k * bw + 2
        d.rectangle([x0, H - m - bh, x0 + bw - 4, H - m], fill=(120, 200, 240))
    d.text((m, 4), "Fish Food completion-time distribution", fill=(220, 230, 240))
    d.text((m, H - m + 8), f"{lo:.0f}s", fill=(180, 195, 210))
    d.text((W - 70, H - m + 8), f"{hi:.0f}s", fill=(180, 195, 210))
    img.save(path)
    print(f"  histogram written to {path}")


def benchmark(
    cfg: Config, runs: int, base_seed: int, plot_path: str | None = None
) -> None:
    hard_note = f", {cfg.hard_fraction:.0%} hard units" if cfg.hard_fraction > 0 else ""
    wk_note = f", {cfg.n_workloads} workloads" if cfg.n_workloads > 1 else ""
    print(
        f"Fish Food benchmark: {runs} run(s), {cfg.n_pellets} pellets{hard_note}"
        f"{wk_note}, dt={cfg.dt}s, cap={cfg.max_seconds}s\n"
    )
    print(f"  {'run':>3}  {'seed':>6}  {'time (s)':>9}  {'mm:ss':>6}  status")
    print("  " + "-" * 44)

    times: list[float] = []
    spreads: list[float] = []
    for i in range(runs):
        seed = base_seed + i
        t0 = time.perf_counter()
        res = run_headless(cfg, seed)
        wall = time.perf_counter() - t0
        secs = res["time"]
        mmss = f"{int(secs) // 60:d}:{int(secs) % 60:02d}"
        status = "done" if res["completed"] else f"CAPPED ({res['remaining']} left)"
        print(
            f"  {i + 1:>3}  {seed:>6}  {secs:>9.2f}  {mmss:>6}  "
            f"{status}  [{wall:.2f}s wall]"
        )
        if res["completed"]:
            times.append(secs)
            spreads.append(res["workload_spread"])

    print("  " + "-" * 44)
    if not times:
        print(
            "\nNo runs completed within the time cap - lower the workload "
            "or raise --max-seconds."
        )
        return

    mean = statistics.mean(times)
    median = statistics.median(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    cv = (stdev / mean * 100.0) if mean else 0.0
    print(f"\nCompletion time over {len(times)} completed run(s):")
    print(f"  mean   = {mean:8.2f} s   ({int(mean) // 60}:{int(mean) % 60:02d})")
    print(f"  median = {median:8.2f} s")
    print(f"  stdev  = {stdev:8.2f} s")
    print(f"  min    = {min(times):8.2f} s")
    print(f"  max    = {max(times):8.2f} s")
    print(f"  spread = {max(times) - min(times):8.2f} s")
    print(f"  CV     = {cv:8.1f} %   (lower = more constant-time)")

    if cfg.n_workloads > 1 and spreads:
        # Fairness gap = how far apart the first and last workload finish within
        # a run. Small = the shared pool clears all workloads together; large =
        # it swarms one workload while others wait.
        mean_gap = statistics.mean(spreads)
        print(
            f"\n  Fairness across {cfg.n_workloads} workloads (per-run gap between"
            f" first & last finish):\n"
            f"    mean gap = {mean_gap:7.2f} s  "
            f"({mean_gap / mean * 100:.0f}% of the total completion time)\n"
            f"    max gap  = {max(spreads):7.2f} s"
        )
        if mean_gap / mean > 0.4:
            print("    => pooling is UNFAIR: some workloads wait a long time.")
        elif mean_gap / mean > 0.15:
            print("    => moderately staggered finishes.")
        else:
            print("    => workloads finish close together (fair sharing).")

    # Verdict: the whole claim is about LOW variance across random layouts.
    if cv < 8:
        verdict = "VERY constant-time (tight clustering)"
    elif cv < 15:
        verdict = "fairly constant-time"
    elif cv < 30:
        verdict = "moderately variable"
    else:
        verdict = "highly variable (not yet constant-time)"
    print(f"  verdict: {verdict}")

    print("\nDistribution:")
    for line in _ascii_histogram(times):
        print(line)

    if plot_path:
        _save_histogram_png(times, plot_path)


def theory_report(cfg: Config, observed: float | None = None) -> None:
    """Print a capacity / queueing estimate WITHOUT running a simulation.

    This computes an *optimistic floor* on completion time assuming perfect
    utilization and zero travel/search, plus the hard-work bottleneck. Real
    runs sit well above this floor; the ratio is the travel/search overhead
    that better routing policies aim to shrink.
    """
    n = cfg.n_pellets
    hard = int(round(n * cfg.hard_fraction))
    easy = n - hard
    easy_bites = easy * 1
    hard_bites = hard * cfg.hard_bites
    total_bites = easy_bites + hard_bites

    print("Fish Food - capacity / queueing estimate (no simulation)\n")
    print(f"  units: {n}  (easy {easy}, hard {hard} x {cfg.hard_bites} bites each)")
    print(f"  work : {total_bites} bites  (easy {easy_bites} + hard {hard_bites})\n")
    print(f"  {'pool':<14}{'count':>6}{'rate/s':>9}{'cap':>9}{'hard?':>7}")
    print("  " + "-" * 45)
    r_total = 0.0
    r_hard = 0.0
    for ft in cfg.fish_types:
        rate = ft.count / ft.cooldown
        r_total += rate
        can_hard = ft.mouth >= cfg.hard_min_mouth
        if can_hard:
            r_hard += rate
        cap = "inf" if ft.max_eat == math.inf else f"{ft.max_eat:g}"
        print(
            f"  {ft.name:<14}{ft.count:>6}{rate:>9.1f}{cap:>9}"
            f"{('yes' if can_hard else 'no'):>7}"
        )
    print("  " + "-" * 45)
    print(f"  total bite-rate (all consumers)   = {r_total:7.1f} bites/s")
    print(f"  hard-capable bite-rate            = {r_hard:7.1f} bites/s\n")

    t_all = total_bites / r_total if r_total else math.inf
    t_hard = (hard_bites / r_hard) if (r_hard and hard_bites) else 0.0
    floor = max(t_all, t_hard)
    print(f"  floor: all work / total rate      = {t_all:7.1f} s")
    if hard_bites > 0:
        tag = "  <-- bottleneck" if t_hard >= t_all else ""
        print(f"  floor: hard work / hard rate      = {t_hard:7.1f} s{tag}")
    print(f"  => optimistic floor               = {floor:7.1f} s")
    print("     (perfect utilization, zero travel/search)")

    if observed:
        util = floor / observed * 100.0 if observed else 0.0
        print(f"\n  observed mean                     = {observed:7.1f} s")
        print(f"  utilization (floor / observed)    = {util:7.1f} %")
        print(f"  travel/search overhead            = {observed / floor:7.1f} x")

    print(
        "\n  Note: caps-over-time and all travel/search are ignored, so this is\n"
        "  a lower bound. The gap between it and real runs is what scheduling\n"
        "  policies (try --policy specialist) aim to close."
    )


# --------------------------------------------------------------------------
# Visual mode (pygame imported lazily here)
# --------------------------------------------------------------------------


def run_visual(cfg: Config, seed: int | None, fps: int, show_graph: bool) -> None:
    import pygame  # lazy: benchmark must work without a display / pygame

    pygame.init()
    pygame.display.set_caption("Fish Food")

    # screen layout
    scale = 150  # pixels per meter
    pad = 20
    pond_px_w = int(cfg.pond_w * scale)
    pond_px_h = int(cfg.pond_h * scale)
    graph_h = 120 if show_graph else 0
    hud_h = 112
    win_w = pond_px_w + 2 * pad
    win_h = pond_px_h + 2 * pad + hud_h + graph_h
    screen = pygame.display.set_mode((win_w, win_h))
    font = pygame.font.SysFont("monospace", 16)
    big = pygame.font.SysFont("monospace", 26, bold=True)
    clock = pygame.time.Clock()

    def to_px(x, y):
        return int(pad + x * scale), int(pad + y * scale)

    WATER = (18, 52, 74)
    WATER_EDGE = (40, 90, 120)
    PELLET = (210, 190, 120)
    STUCK = (150, 130, 80)
    HARD = (225, 95, 115)  # "hard" work units (only big consumers can finish)
    WK_PALETTE = [
        (210, 190, 120),
        (130, 200, 150),
        (150, 180, 235),
        (225, 175, 120),
        (200, 150, 220),
        (130, 205, 205),
    ]

    # Mutable, in-window settings the user can change with keys (see below).
    s_seed = seed if seed is not None else random.randrange(1_000_000)
    s_workloads = cfg.n_workloads
    s_hard = cfg.hard_fraction
    s_staggered = cfg.staggered_arrivals

    def make_sim():
        live_cfg = replace(
            cfg,
            n_workloads=s_workloads,
            hard_fraction=s_hard,
            staggered_arrivals=s_staggered,
        )
        return live_cfg, Simulation(live_cfg, seed=s_seed)

    live_cfg, sim = make_sim()
    total = cfg.n_pellets
    history: list[int] = [sim.alive_count]
    paused = False
    finished_at: float | None = None
    running = True

    def fmtk(v: int) -> str:
        """Compact count: 3000 -> '3.0K', keeps the HUD narrow."""
        return f"{v / 1000:.1f}K" if v >= 1000 else str(v)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                else:
                    changed = True
                    if k == pygame.K_r:  # new random run
                        s_seed = random.randrange(1_000_000)
                    elif k == pygame.K_n:  # replay the same seed
                        pass
                    elif k == pygame.K_RIGHTBRACKET:  # more workloads
                        s_workloads = min(6, s_workloads + 1)
                    elif k == pygame.K_LEFTBRACKET:  # fewer workloads
                        s_workloads = max(1, s_workloads - 1)
                    elif k == pygame.K_h:  # toggle hard-unit fraction
                        s_hard = 0.0 if s_hard > 0 else 0.25
                    elif k == pygame.K_a:  # toggle staggered arrivals
                        s_staggered = not s_staggered
                    else:
                        changed = False
                    if changed:
                        live_cfg, sim = make_sim()
                        history = [sim.alive_count]
                        finished_at = None
                        paused = False

        if not paused and not sim.done:
            sim.step()
            history.append(sim.alive_count)
            if sim.done and finished_at is None:
                finished_at = sim.t
        elif sim.done and finished_at is None:
            finished_at = sim.t

        # ---- draw ----
        screen.fill((12, 24, 34))

        # pond
        pond_rect = pygame.Rect(pad, pad, pond_px_w, pond_px_h)
        pygame.draw.rect(screen, WATER, pond_rect)
        pygame.draw.rect(screen, WATER_EDGE, pond_rect, 3)

        # lily pads
        for lx, ly, lr in cfg.lily_pads:
            c = to_px(lx, ly)
            pygame.draw.circle(screen, (34, 80, 52), c, int(lr * scale))
            pygame.draw.circle(screen, (52, 110, 70), c, int(lr * scale), 2)

        # pump
        pc = to_px(*cfg.pump_pos)
        pygame.draw.circle(screen, (60, 130, 160), pc, int(cfg.pump_radius * scale), 1)
        pygame.draw.circle(screen, (90, 180, 210), pc, 4)

        # pellets (only those that have dropped into the water)
        present = sim.present_mask
        if present.any():
            pts = sim.p_pos[present]
            stuck = sim.p_stuck[present]
            hard = sim.p_minmouth[present] > 0
            wids = sim.p_workload[present]
            multi = sim.n_wk > 1
            for (x, y), st, hd, wid in zip(pts, stuck, hard, wids):
                if st:
                    col = STUCK
                elif hd:
                    col = HARD
                elif multi:
                    col = WK_PALETTE[int(wid) % len(WK_PALETTE)]
                else:
                    col = PELLET
                pygame.draw.circle(screen, col, to_px(x, y), 3 if hd else 2)

        # fish
        for i in range(sim.f_pos.shape[0]):
            x, y = sim.f_pos[i]
            col = tuple(int(c) for c in sim.f_color[i])
            r = max(2, int(sim.f_drawsize[i] * scale))
            cx, cy = to_px(x, y)
            pygame.draw.circle(screen, col, (cx, cy), r)
            # little tail in the direction of travel
            v = sim.f_vel[i]
            n = math.hypot(v[0], v[1])
            if n > 1e-6 and r >= 4:
                tx = cx - int(v[0] / n * r * 1.6)
                ty = cy - int(v[1] / n * r * 1.6)
                pygame.draw.line(screen, col, (cx, cy), (tx, ty), max(1, r // 3))

        # HUD
        hud_y = pad + pond_px_h + 8
        eaten = total - sim.alive_count
        secs = sim.t
        mmss = f"{int(secs) // 60}:{int(secs) % 60:02d}"
        # Line 1: running clock, or the DONE banner once finished.
        if finished_at is not None:
            fmmss = f"{int(finished_at) // 60}:{int(finished_at) % 60:02d}"
            line1 = big.render(
                f"DONE in {fmmss}  ({finished_at:.1f}s)", True, (120, 240, 150)
            )
        else:
            line1 = big.render(f"t = {mmss}   ({secs:5.1f}s)", True, (230, 230, 230))
        screen.blit(line1, (pad, hud_y))
        # Line 2: compact work counts.
        screen.blit(
            font.render(
                f"work: {fmtk(sim.alive_count)} left / {fmtk(eaten)} done /"
                f" {fmtk(total)} total",
                True,
                (200, 210, 220),
            ),
            (pad, hud_y + 30),
        )
        # Line 3: live settings readout.
        hard_txt = f"{int(s_hard * 100)}%" if s_hard > 0 else "off"
        arr_txt = "staggered" if s_staggered else "burst"
        screen.blit(
            font.render(
                f"seed {s_seed}   workloads {s_workloads}   "
                f"hard {hard_txt}   arrivals {arr_txt}",
                True,
                (150, 180, 205),
            ),
            (pad, hud_y + 50),
        )
        # Line 4: key hints.
        screen.blit(
            font.render(
                "[space] pause   [R] random   [N] replay   [ [ / ] ] workloads   "
                "[H] hard   [A] arrivals   [Q] quit",
                True,
                (120, 150, 170),
            ),
            (pad, hud_y + 70),
        )

        # fish-type legend (right side, clear of the text columns)
        legend_x = win_w - 250
        for i, ft in enumerate(cfg.fish_types):
            mask = sim.f_type == i
            eaten_t = int(sim.f_eaten[mask].sum())
            pygame.draw.circle(screen, ft.color, (legend_x + 6, hud_y + 8 + i * 18), 6)
            screen.blit(
                font.render(
                    f"{ft.name:<13} x{ft.count:<3} {fmtk(eaten_t)}",
                    True,
                    (200, 200, 200),
                ),
                (legend_x + 18, hud_y + i * 18),
            )

        if paused:
            screen.blit(
                big.render("PAUSED", True, (250, 220, 120)),
                (pad + pond_px_w // 2 - 55, pad + 8),
            )

        # live "pellets remaining" graph
        if show_graph:
            gx, gy = pad, pad + pond_px_h + hud_h
            gw, gh = pond_px_w, graph_h - 16
            pygame.draw.rect(screen, (20, 30, 40), (gx, gy, gw, gh))
            pygame.draw.rect(screen, (60, 80, 100), (gx, gy, gw, gh), 1)
            if len(history) > 1:
                n = len(history)
                step = max(1, n // gw)
                samples = history[::step]
                ppts = []
                for k, val in enumerate(samples):
                    px = gx + int(k / max(1, len(samples) - 1) * gw)
                    py = gy + gh - int(val / total * gh)
                    ppts.append((px, py))
                if len(ppts) > 1:
                    pygame.draw.lines(screen, (120, 200, 240), False, ppts, 2)
            screen.blit(
                font.render("pellets remaining vs time", True, (120, 150, 170)),
                (gx + 6, gy + 4),
            )

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_config(args) -> Config:
    cfg = Config()
    if args.pellets is not None:
        cfg.n_pellets = args.pellets
    if args.max_seconds is not None:
        cfg.max_seconds = args.max_seconds
    if args.hard_fraction is not None:
        cfg.hard_fraction = args.hard_fraction
    if args.workloads is not None:
        cfg.n_workloads = args.workloads
    if args.staggered:
        cfg.staggered_arrivals = True
    if args.arrival_interval is not None:
        cfg.arrival_interval = args.arrival_interval
    if args.policy is not None:
        cfg.policy = args.policy
    if args.legacy_search:
        cfg.recruit = False
        cfg.ars = False
    # Search-behavior tuning knobs (for predictability sweeps).
    if args.recruit_radius is not None:
        cfg.recruit_radius = args.recruit_radius
    if args.recruit_recent is not None:
        cfg.recruit_recent = args.recruit_recent
    if args.ars_full_turn is not None:
        cfg.ars_full_turn = args.ars_full_turn
    if args.ars_lost_turn is not None:
        cfg.ars_lost_turn = args.ars_lost_turn
    if args.ars_memory is not None:
        cfg.ars_memory = args.ars_memory
    return cfg


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Fish Food - watch a pond eat a batch of pellets, or "
        "benchmark how constant the completion time is."
    )
    p.add_argument(
        "--runs",
        type=int,
        default=0,
        help="headless benchmark: run N sims with no window and "
        "report completion-time statistics",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="base random seed (benchmark uses seed, seed+1, ...)",
    )
    p.add_argument(
        "--pellets", type=int, default=None, help="override the number of pellets"
    )
    p.add_argument(
        "--hard-fraction",
        type=float,
        default=None,
        help="fraction of units that are 'hard' (multi-bite, big-mouth-only); "
        "models a workload of mixed difficulty",
    )
    p.add_argument(
        "--workloads",
        type=int,
        default=None,
        help="number of concurrent workloads (separate clumps) sharing the one "
        "agent pool; reports a fairness gap between first & last to finish",
    )
    p.add_argument(
        "--staggered",
        action="store_true",
        help="with >1 workload, workloads arrive over time (a queue) instead of "
        "all at once",
    )
    p.add_argument(
        "--arrival-interval",
        type=float,
        default=None,
        help="seconds between staggered workload arrivals (with --staggered)",
    )
    p.add_argument(
        "--plot",
        type=str,
        default=None,
        help="benchmark: save a completion-time histogram PNG to this path",
    )
    p.add_argument(
        "--policy",
        choices=("greedy", "specialist"),
        default=None,
        help="scheduling policy: 'greedy' (default) or 'specialist' "
        "(big consumers prefer hard units)",
    )
    p.add_argument(
        "--legacy-search",
        action="store_true",
        help="disable recruitment + area-restricted search (baseline blind "
        "wandering, for A/B comparison)",
    )
    # Search-behavior tuning knobs (sweep these to minimize CV).
    p.add_argument(
        "--recruit-radius",
        type=float,
        default=None,
        help="how far idle fish are pulled toward a feeder (m)",
    )
    p.add_argument(
        "--recruit-recent",
        type=float,
        default=None,
        help="seconds a fish counts as 'feeding' after a bite",
    )
    p.add_argument(
        "--ars-full-turn",
        type=float,
        default=None,
        help="wander turn strength just after eating (local search)",
    )
    p.add_argument(
        "--ars-lost-turn",
        type=float,
        default=None,
        help="wander turn strength when long unfed (ballistic)",
    )
    p.add_argument(
        "--ars-memory",
        type=float,
        default=None,
        help="seconds to ramp from local search to dispersal",
    )
    p.add_argument(
        "--theory",
        action="store_true",
        help="print a capacity/queueing estimate (no simulation) and exit",
    )
    p.add_argument(
        "--observed",
        type=float,
        default=None,
        help="with --theory: an observed mean (s) to compare against the floor",
    )
    p.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="safety cap on simulated time per run",
    )
    p.add_argument(
        "--fps", type=int, default=60, help="visual mode target frames per second"
    )
    p.add_argument(
        "--no-graph",
        action="store_true",
        help="visual mode: hide the pellets-remaining graph",
    )
    args = p.parse_args(argv)

    cfg = build_config(args)

    if args.theory:
        theory_report(cfg, observed=args.observed)
        return 0

    if args.runs and args.runs > 0:
        benchmark(cfg, args.runs, args.seed, plot_path=args.plot)
        return 0

    try:
        run_visual(cfg, seed=args.seed, fps=args.fps, show_graph=not args.no_graph)
    except Exception as exc:  # pragma: no cover - visual/display issues
        print(f"Visual mode could not start ({exc}).", file=sys.stderr)
        print(
            "Try the headless benchmark instead:  python fish_food.py --runs 5",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
