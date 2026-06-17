#!/usr/bin/env python3
# Fish Food - https://github.com/silversummitco/fishfood
# Copyright (c) 2026 SilverSummitCo LLC.  Licensed under the PolyForm
# Noncommercial License 1.0.0 - see the LICENSE file (commercial use:
# COMMERCIAL.md).
# Required Notice: Copyright (c) 2026 SilverSummitCo LLC
# (https://github.com/silversummitco/fishfood)
"""Render a Fish Food run to an animated GIF (headless, no display needed).

Reuses the Simulation from fish_food.py and draws each sampled frame with
Pillow, so it works anywhere - no pygame, no window, no GPU.

    python render_gif.py --seed 7 --out fishfood.gif
"""

from __future__ import annotations

import argparse

from PIL import Image, ImageDraw

from fish_food import Config, Simulation


def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def render(
    cfg: Config,
    seed: int,
    out: str,
    scale: int,
    every: int,
    ms_per_frame: int,
    max_frames: int,
    hold_frames: int,
) -> None:
    sim = Simulation(cfg, seed=seed)
    total = cfg.n_pellets

    pad = 16
    hud = 56
    W = int(cfg.pond_w * scale) + 2 * pad
    H = int(cfg.pond_h * scale) + 2 * pad + hud

    def px(x, y):
        return pad + x * scale, pad + y * scale

    WATER_TOP = (22, 64, 92)
    WATER_BOT = (12, 40, 60)
    EDGE = (52, 110, 140)

    # Pre-build a vertical water gradient background.
    bg = Image.new("RGB", (W, H), (8, 18, 26))
    bgd = ImageDraw.Draw(bg)
    pond_h_px = int(cfg.pond_h * scale)
    for row in range(pond_h_px):
        t = row / max(1, pond_h_px - 1)
        bgd.line(
            [(pad, pad + row), (W - pad, pad + row)], fill=lerp(WATER_TOP, WATER_BOT, t)
        )
    bgd.rectangle([pad, pad, W - pad, pad + pond_h_px], outline=EDGE, width=3)
    for lx, ly, lr in cfg.lily_pads:
        cx, cy = px(lx, ly)
        r = lr * scale
        bgd.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(30, 74, 48),
            outline=(48, 104, 66),
            width=2,
        )
    pcx, pcy = px(*cfg.pump_pos)
    pr = cfg.pump_radius * scale
    bgd.ellipse(
        [pcx - pr, pcy - pr, pcx + pr, pcy + pr], outline=(54, 120, 150), width=1
    )

    try:
        from PIL import ImageFont

        font = ImageFont.load_default()
    except Exception:
        font = None

    frames: list[Image.Image] = []
    max_steps = int(cfg.max_seconds / cfg.dt)
    step = 0
    finished_at = None

    def snapshot():
        img = bg.copy()
        d = ImageDraw.Draw(img)
        present = sim.present_mask
        if present.any():
            pts = sim.p_pos[present]
            stuck = sim.p_stuck[present]
            hard = sim.p_minmouth[present] > 0
            for (x, y), st, hd in zip(pts, stuck, hard):
                sx, sy = px(x, y)
                if st:
                    col, rad = (150, 130, 80), 1.6
                elif hd:
                    col, rad = (225, 95, 115), 2.4
                else:
                    col, rad = (224, 200, 130), 1.6
                d.ellipse([sx - rad, sy - rad, sx + rad, sy + rad], fill=col)
        for i in range(sim.f_pos.shape[0]):
            x, y = sim.f_pos[i]
            sx, sy = px(x, y)
            col = tuple(int(c) for c in sim.f_color[i])
            r = max(2.0, sim.f_drawsize[i] * scale)
            vx, vy = sim.f_vel[i]
            n = (vx * vx + vy * vy) ** 0.5
            if n > 1e-6 and r >= 3:
                tx, ty = sx - vx / n * r * 1.7, sy - vy / n * r * 1.7
                d.line([(sx, sy), (tx, ty)], fill=col, width=max(1, int(r / 2)))
            d.ellipse([sx - r, sy - r, sx + r, sy + r], fill=col)
        secs = sim.t
        mmss = f"{int(secs) // 60}:{int(secs) % 60:02d}"
        eaten = total - sim.alive_count
        bar_w = W - 2 * pad
        frac = eaten / total
        by = pad + pond_h_px + 12
        d.rectangle(
            [pad, by, pad + bar_w, by + 12], fill=(28, 40, 52), outline=(60, 80, 100)
        )
        d.rectangle([pad, by, pad + int(bar_w * frac), by + 12], fill=(120, 200, 240))
        label = f"t {mmss}   eaten {eaten}/{total}   left {sim.alive_count}"
        if finished_at is not None:
            fm = f"{int(finished_at) // 60}:{int(finished_at) % 60:02d}"
            label = f"DONE in {fm} ({finished_at:.1f}s)   eaten {total}/{total}"
        if font is not None:
            d.text((pad, by + 18), label, fill=(210, 220, 230), font=font)
        return img

    while step < max_steps and len(frames) < max_frames:
        if step % every == 0:
            frames.append(snapshot())
        if sim.done:
            if finished_at is None:
                finished_at = sim.t
            break
        sim.step()
        if sim.done and finished_at is None:
            finished_at = sim.t
        step += 1

    frames.append(snapshot())  # final frame
    frames.extend([frames[-1]] * hold_frames)  # hold on the finished pond

    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=ms_per_frame,
        loop=0,
        optimize=True,
    )
    print(
        f"Wrote {out}: {len(frames)} frames, {W}x{H}, "
        f"finished at {finished_at:.1f}s sim time"
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Render a Fish Food run to a GIF.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="fishfood.gif")
    p.add_argument("--scale", type=int, default=110, help="pixels per meter")
    p.add_argument("--every", type=int, default=4, help="capture every Nth sim step")
    p.add_argument("--ms", type=int, default=60, help="ms per GIF frame")
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument(
        "--hold", type=int, default=18, help="extra frames on the final image"
    )
    p.add_argument(
        "--pellets",
        type=int,
        default=None,
        help="override pellet count (e.g. a smaller value for a short demo clip)",
    )
    p.add_argument(
        "--hard-fraction",
        type=float,
        default=None,
        help="fraction of 'hard' units (multi-bite, big-mouth-only)",
    )
    args = p.parse_args()

    cfg = Config()
    if args.pellets is not None:
        cfg.n_pellets = args.pellets
    if args.hard_fraction is not None:
        cfg.hard_fraction = args.hard_fraction
    render(
        cfg,
        args.seed,
        args.out,
        args.scale,
        args.every,
        args.ms,
        args.max_frames,
        args.hold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
