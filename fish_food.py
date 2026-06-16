#!/usr/bin/env python3
# Fish Food - https://github.com/silversummitco/fishfood
# Copyright (c) 2026 Silver Summit Co.  Licensed under the Fish Food Community
# License (FFCL) v1.0 - see the LICENSE file. Free for non-commercial use with
# attribution; commercial/profit use requires permission and a profit-share.
"""Fish Food - an agent-based pond-feeding simulator.

The idea (see AGENTS.md for the full story): a fixed handful of fish-food
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
import statistics
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Units are meters and seconds. Defaults are the "first guesses" from
# AGENTS.md; everything here is meant to be tuned.


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
    def _init_pellets(self) -> None:
        cfg = self.cfg
        n = cfg.n_pellets
        cx, cy = cfg.pond_w / 2.0, cfg.pond_h / 2.0

        # Tight clump near the center (uniform over a small disc) - the food
        # fills the whole basketball-sized circle, not just its rim.
        r = cfg.clump_radius * np.sqrt(self.rng.random(n))
        a = self.rng.random(n) * 2.0 * np.pi
        self.p_pos = np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)
        self.p_vel = np.zeros((n, 2))
        self.p_alive = np.ones(n, dtype=bool)
        self.p_stuck = np.zeros(n, dtype=bool)
        # Pellets appear progressively, so the clump visibly fills up over the
        # first couple of seconds rather than popping in all at once.
        self.p_drop_t = self.rng.random(n) * cfg.drop_window

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

        # nearest alive pellet within sense radius, per fish
        within = dist < self.f_sense[:, None]
        masked = np.where(within, dist, np.inf)  # (M, Na)
        nearest = np.argmin(masked, axis=1)  # (M,)
        has_target = np.isfinite(masked[np.arange(M), nearest])

        # direction toward target (diff is pellet - fish, i.e. toward pellet)
        tgt_off = diff[np.arange(M), nearest]  # (M, 2)
        tgt_dist = dist[np.arange(M), nearest][:, None]
        tgt_dir = np.where(tgt_dist > 1e-9, tgt_off / np.maximum(tgt_dist, 1e-9), 0.0)

        # wander: heading does a small random walk
        self.f_wander += self.rng.normal(0.0, 0.6, M) * 0.15
        wander_dir = np.stack([np.cos(self.f_wander), np.sin(self.f_wander)], axis=1)

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

        # eligible[m, j] : fish m may eat alive-pellet j (within its mouth)
        eligible = (dist < self.f_mouth[:, None]) & ready[:, None]
        if not eligible.any():
            return

        # Resolve greedily over the (few) ready+eligible fish. Closest first
        # so a pellet goes to the hungriest/nearest mouth.
        masked = np.where(eligible, dist, np.inf)
        cand_j = np.argmin(masked, axis=1)  # nearest pellet per fish
        cand_d = masked[np.arange(masked.shape[0]), cand_j]
        biters = np.where(np.isfinite(cand_d))[0]
        # nearest mouths get first pick (avoids two fish "eating" one pellet)
        biters = biters[np.argsort(cand_d[biters])]

        eaten_local = set()
        for m in biters:
            j = int(cand_j[m])
            if j in eaten_local:
                continue
            eaten_local.add(j)
            pellet = int(ai[j])
            self.p_alive[pellet] = False
            self.f_eaten[m] += 1
            self.f_cool_timer[m] = self.f_cooldown[m]


# --------------------------------------------------------------------------
# Headless benchmark
# --------------------------------------------------------------------------


def run_headless(cfg: Config, seed: int | None) -> dict:
    """Run one sim to completion with no window. Returns summary stats."""
    sim = Simulation(cfg, seed=seed)
    max_steps = int(cfg.max_seconds / cfg.dt)
    while not sim.done and sim.steps < max_steps:
        sim.step()
    return {
        "seed": seed,
        "time": sim.t,
        "steps": sim.steps,
        "remaining": sim.alive_count,
        "completed": sim.done,
    }


def benchmark(cfg: Config, runs: int, base_seed: int) -> None:
    print(
        f"Fish Food benchmark: {runs} run(s), {cfg.n_pellets} pellets, "
        f"dt={cfg.dt}s, cap={cfg.max_seconds}s\n"
    )
    print(f"  {'run':>3}  {'seed':>6}  {'time (s)':>9}  {'mm:ss':>6}  status")
    print("  " + "-" * 44)

    times: list[float] = []
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

    print("  " + "-" * 44)
    if not times:
        print(
            "\nNo runs completed within the time cap - lower the workload "
            "or raise --max-seconds."
        )
        return

    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    cv = (stdev / mean * 100.0) if mean else 0.0
    print(f"\nCompletion time over {len(times)} completed run(s):")
    print(f"  mean   = {mean:8.2f} s   ({int(mean) // 60}:{int(mean) % 60:02d})")
    print(f"  stdev  = {stdev:8.2f} s")
    print(f"  min    = {min(times):8.2f} s")
    print(f"  max    = {max(times):8.2f} s")
    print(f"  spread = {max(times) - min(times):8.2f} s")
    print(f"  CV     = {cv:8.1f} %   (lower = more constant-time)")


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
    hud_h = 80
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

    sim = Simulation(cfg, seed=seed)
    history: list[int] = [sim.alive_count]
    total = cfg.n_pellets
    paused = False
    finished_at: float | None = None
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim = Simulation(cfg, seed=None)
                    history = [sim.alive_count]
                    finished_at = None

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
            for (x, y), st in zip(pts, stuck):
                col = STUCK if st else PELLET
                pygame.draw.circle(screen, col, to_px(x, y), 2)

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
        screen.blit(
            big.render(f"t = {mmss}   ({secs:5.1f}s)", True, (230, 230, 230)),
            (pad, hud_y),
        )
        screen.blit(
            font.render(
                f"pellets: {sim.alive_count:4d} left / {eaten:4d} eaten / {total} total",
                True,
                (200, 210, 220),
            ),
            (pad, hud_y + 32),
        )

        # fish-type legend with live eaten counts
        legend_x = pad + 360
        for i, ft in enumerate(cfg.fish_types):
            mask = sim.f_type == i
            eaten_t = int(sim.f_eaten[mask].sum())
            pygame.draw.circle(screen, ft.color, (legend_x + 8, hud_y + 8 + i * 18), 6)
            screen.blit(
                font.render(
                    f"{ft.name:<13} x{ft.count:<3}  ate {eaten_t}",
                    True,
                    (200, 200, 200),
                ),
                (legend_x + 22, hud_y + i * 18),
            )

        if finished_at is not None:
            done_txt = big.render(
                f"DONE in {int(finished_at) // 60}:{int(finished_at) % 60:02d}  "
                f"({finished_at:.1f}s)   [R = new run]",
                True,
                (120, 240, 150),
            )
            screen.blit(done_txt, (pad, hud_y - 2))
        if paused:
            screen.blit(
                font.render("PAUSED  [space]", True, (250, 220, 120)),
                (win_w - 170, hud_y),
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

    if args.runs and args.runs > 0:
        benchmark(cfg, args.runs, args.seed)
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
