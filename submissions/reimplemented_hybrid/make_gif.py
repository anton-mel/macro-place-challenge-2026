"""
make_gif.py — animate yale_am's smooth-proxy GD stage.
yale_am (Anton Melnychuk).

Real optimizer snapshots (GD steps + legalization), stitched to GIF.

Usage:
    python submissions/reimplemented_hybrid/make_gif.py
    GIF_BENCH=ibm10 GIF_STEPS=400 python submissions/reimplemented_hybrid/make_gif.py
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import imageio.v2 as imageio

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _THIS_DIR)

from macro_place.loader import load_benchmark_from_dir

import smooth_proxy as sp
from fast_proxy import FastProxy
from placer import _SmoothGD, _legalize_hard_macros

NAME = os.environ.get("GIF_BENCH", "ibm01")
N_STEPS = int(os.environ.get("GIF_STEPS", "650"))
MAX_SEC = float(os.environ.get("GIF_MAXSEC", "900"))
N_FRAMES = int(os.environ.get("GIF_FRAMES", "50"))
OUT = os.environ.get("GIF_OUT", os.path.join(_REPO, "submissions", "reimplemented_hybrid",
                                              f"yale_am_{NAME}.gif"))

bench_dir = os.path.join(_REPO, "external", "MacroPlacement", "Testcases", "ICCAD04", NAME)
print(f"[gif] loading {NAME} from {bench_dir}")
benchmark, plc = load_benchmark_from_dir(bench_dir)
fp = FastProxy(benchmark)

gd = _SmoothGD(benchmark, density_w=0.7, congestion_w=1.3)
nh = benchmark.num_hard_macros
sizes = benchmark.macro_sizes.cpu().numpy().astype(np.float64)
canvas_w, canvas_h = float(benchmark.canvas_width), float(benchmark.canvas_height)

# denser snapshots early where positions move the most
snap_steps = set(int(s) for s in np.unique(np.linspace(0, N_STEPS - 1, N_FRAMES).astype(int)))

torch.manual_seed(42)
init = torch.tensor(benchmark.macro_positions.numpy().astype(np.float64), dtype=torch.float32)
init[:, 0].clamp_(gd.half_w, gd.canvas_w - gd.half_w)
init[:, 1].clamp_(gd.half_h, gd.canvas_h - gd.half_h)
fixed_xy = init.clone()
pos = torch.nn.Parameter(init.clone())
opt = torch.optim.Adam([pos], lr=0.2)
gamma_start, gamma_end, lr0, lr1 = 0.5, 8.0, 0.2, 0.02
anneal_split = 0.5

frames = []  # (label, position_snapshot)
t0 = time.time()
print(f"[gif] running {N_STEPS} GD steps, snapshotting {len(snap_steps)} frames")
for step in range(N_STEPS):
    if step > 0 and step % 5 == 0 and time.time() - t0 > MAX_SEC:
        print(f"[gif] wall cap hit at step {step}")
        break
    t = step / max(N_STEPS - 1, 1)
    if t < anneal_split:
        gamma = gamma_start + t / anneal_split * (2.0 - gamma_start)
    else:
        gamma = 2.0 * (gamma_end / 2.0) ** ((t - anneal_split) / (1 - anneal_split))
    lr = (lr0 * (step + 1) / 20 if step < 20
          else lr0 + (step - 20) / max(N_STEPS - 21, 1) * (lr1 - lr0))
    for g in opt.param_groups:
        g["lr"] = lr
    opt.zero_grad()
    pos_aug = torch.cat([pos, gd.port_pos], dim=0)
    wl, d, c = sp.smooth_costs(
        pos_aug, benchmark.macro_sizes.float(), gd.owner, gd.offset, gd.mask,
        gd.net_weight, gd.net_count, gd.grid_cols, gd.grid_rows,
        gd.canvas_w, gd.canvas_h, gd.h_tracks, gd.v_tracks,
        h_alloc=1.0, v_alloc=1.0, n_hard=gd.n_hard, n_macros=gd.n_macros, gamma_wl=gamma)
    (wl + gd.dw * d + gd.cw_ * c).backward()
    opt.step()
    with torch.no_grad():
        pos[~gd.movable] = fixed_xy[~gd.movable]
        pos[:, 0].clamp_(gd.half_w, gd.canvas_w - gd.half_w)
        pos[:, 1].clamp_(gd.half_h, gd.canvas_h - gd.half_h)
    if step in snap_steps:
        frames.append((f"GD step {step + 1}/{N_STEPS}  (gamma={gamma:.2f})",
                       pos.detach().numpy().astype(np.float64)))

gd_final = pos.detach().numpy().astype(np.float64)
print(f"[gif] GD done in {time.time() - t0:.0f}s, legalizing...")
legal_pos, overlaps = _legalize_hard_macros(gd_final, benchmark)
print(f"[gif] legalized, overlaps={overlaps}")
frames.append(("legalize (hard-macro overlaps resolved)", legal_pos))

# ── render ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(7.6, 8.0), dpi=110)
images = []
half_w, half_h = sizes[:, 0] * 0.5, sizes[:, 1] * 0.5

# color hard macros by area rank so the colormap isn't flattened by one
# giant macro: small -> blue/teal, mid -> green/yellow, large -> orange/red
area = sizes[:nh, 0] * sizes[:nh, 1]
rank01 = np.argsort(np.argsort(area)) / max(nh - 1, 1)
hard_colors = plt.cm.turbo(0.04 + 0.92 * rank01)

print(f"[gif] rendering {len(frames)} frames")
for label, p in frames:
    fig.clf()
    ax = fig.add_axes([0.06, 0.05, 0.90, 0.86])
    ax.set_xlim(0, canvas_w); ax.set_ylim(0, canvas_h); ax.set_aspect("equal")
    ax.add_patch(Rectangle((0, 0), canvas_w, canvas_h, fill=False, edgecolor="black", lw=2))
    ax.set_xticks([]); ax.set_yticks([])

    soft_patches = [Rectangle((p[i, 0] - sizes[i, 0] / 2, p[i, 1] - sizes[i, 1] / 2),
                              sizes[i, 0], sizes[i, 1]) for i in range(nh, benchmark.num_macros)]
    ax.add_collection(PatchCollection(soft_patches, facecolor="#9db8d8",
                                      alpha=0.12, edgecolor="none", zorder=1))
    hard_patches = [Rectangle((p[i, 0] - half_w[i], p[i, 1] - half_h[i]),
                              sizes[i, 0], sizes[i, 1]) for i in range(nh)]
    ax.add_collection(PatchCollection(hard_patches, facecolor=hard_colors, alpha=0.92,
                                      edgecolor="#1b2838", linewidths=0.4, zorder=3))

    cost = fp.proxy(p)
    fig.suptitle(f"{NAME} · yale_am", fontsize=13, fontweight="bold", y=0.975)
    ax.set_title(f"{label}    |    fast proxy ≈ {cost:.3f}", fontsize=11, pad=8)

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    images.append(buf[:, :, :3].copy())

images = [images[0]] * 6 + images + [images[-1]] * 14
print(f"[gif] writing {OUT}  ({len(images)} frames)")
imageio.mimsave(OUT, images, duration=0.12, loop=0)
print(f"[gif] done: {OUT}")
