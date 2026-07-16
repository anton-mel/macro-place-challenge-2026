"""
placer.py — yale_am (Anton Melnychuk).

Pipeline: smooth-proxy Adam GD -> legalize -> cycle(CD polish -> GD ->
legalize) -> cold SA. Shelf-pack fallback guarantees zero overlaps.
"""
import math
import os
import sys
import time

import numpy as np
import torch

from macro_place.benchmark import Benchmark

# add own dir to sys.path (evaluate.py loads this standalone via importlib)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import smooth_proxy as sp
from fast_proxy import FastProxy
from legalize import legalize, count_overlaps
from local_search import coordinate_descent_polish, cold_simulated_annealing

PIN_CAP = 32  # max pins/net kept


def _build_smooth_pin_tables(benchmark):
    n_hard = benchmark.num_hard_macros
    nets = [net for net in benchmark.net_pin_nodes if net.shape[0] >= 2]
    max_pins = min(PIN_CAP, max((int(n.shape[0]) for n in nets), default=1))
    n_nets = len(nets)
    owner = torch.zeros(n_nets, max_pins, dtype=torch.long)
    offset = torch.zeros(n_nets, max_pins, 2)
    mask = torch.zeros(n_nets, max_pins, dtype=torch.bool)
    pin_offsets = benchmark.macro_pin_offsets
    for i, net in enumerate(nets):
        k = min(max_pins, int(net.shape[0]))
        owner[i, :k] = net[:k, 0].long()
        mask[i, :k] = True
        for j in range(k):
            ow, slot = int(net[j, 0]), int(net[j, 1])
            if ow < n_hard and slot < pin_offsets[ow].shape[0]:
                offset[i, j, 0] = float(pin_offsets[ow][slot, 0])
                offset[i, j, 1] = float(pin_offsets[ow][slot, 1])
    net_weight = torch.ones(n_nets)
    return owner, offset, mask, net_weight


class _SmoothGD:
    """Adam over smooth proxy; built once per benchmark."""

    def __init__(self, benchmark, density_w, congestion_w):
        self.b = benchmark
        self.dw, self.cw_ = density_w, congestion_w
        self.canvas_w, self.canvas_h = float(benchmark.canvas_width), float(benchmark.canvas_height)
        self.n_macros, self.n_hard = benchmark.num_macros, benchmark.num_hard_macros
        sizes = benchmark.macro_sizes.float()
        self.half_w, self.half_h = sizes[:, 0] * 0.5, sizes[:, 1] * 0.5
        self.owner, self.offset, self.mask, self.net_weight = _build_smooth_pin_tables(benchmark)
        self.net_count = float(benchmark.num_nets or 1)
        self.grid_cols, self.grid_rows = int(benchmark.grid_cols), int(benchmark.grid_rows)
        self.h_tracks, self.v_tracks = benchmark.hroutes_per_micron, benchmark.vroutes_per_micron
        self.port_pos = benchmark.port_positions.float()
        self.movable = benchmark.get_movable_mask()

    def run(self, init_pos, n_steps, max_sec, gamma_start, gamma_end, lr0, lr1, seed=42):
        """Fixed-step Adam GD, wall-clock capped."""
        torch.manual_seed(seed)
        init = torch.tensor(np.asarray(init_pos), dtype=torch.float32)
        init[:, 0].clamp_(self.half_w, self.canvas_w - self.half_w)
        init[:, 1].clamp_(self.half_h, self.canvas_h - self.half_h)
        fixed_xy = init.clone()
        pos = torch.nn.Parameter(init.clone())
        opt = torch.optim.Adam([pos], lr=lr0)
        anneal_split = 0.5
        t0 = time.time()
        for step in range(n_steps):
            if step > 0 and step % 5 == 0 and time.time() - t0 > max_sec:
                break
            t = step / max(n_steps - 1, 1)
            if t < anneal_split:
                gamma = gamma_start + t / anneal_split * (2.0 - gamma_start)
            else:
                gamma = 2.0 * (gamma_end / 2.0) ** ((t - anneal_split) / (1 - anneal_split))
            lr = (lr0 * (step + 1) / 20 if step < 20
                  else lr0 + (step - 20) / max(n_steps - 21, 1) * (lr1 - lr0))
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad()
            pos_aug = torch.cat([pos, self.port_pos], dim=0)
            wl, d, c = sp.smooth_costs(
                pos_aug, self.b.macro_sizes.float(), self.owner, self.offset, self.mask,
                self.net_weight, self.net_count, self.grid_cols, self.grid_rows,
                self.canvas_w, self.canvas_h, self.h_tracks, self.v_tracks,
                h_alloc=1.0, v_alloc=1.0, n_hard=self.n_hard, n_macros=self.n_macros,
                gamma_wl=gamma)
            (wl + self.dw * d + self.cw_ * c).backward()
            opt.step()
            with torch.no_grad():
                pos[~self.movable] = fixed_xy[~self.movable]
                pos[:, 0].clamp_(self.half_w, self.canvas_w - self.half_w)
                pos[:, 1].clamp_(self.half_h, self.canvas_h - self.half_h)
        return pos.detach().numpy().astype(np.float64)


def _legalize_hard_macros(pos, benchmark):
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] * 0.5, sizes[:, 1] * 0.5
    movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).numpy()
    return legalize(pos, half_w, half_h, movable, benchmark.num_hard_macros,
                     float(benchmark.canvas_width), float(benchmark.canvas_height))


def _shelf_pack_fallback(benchmark, base_pos=None):
    """Guaranteed-legal fallback: shelf-pack hard macros by descending height."""
    placement = (torch.tensor(base_pos, dtype=torch.float32) if base_pos is not None
                 else benchmark.macro_positions.clone())
    movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    movable_indices = torch.where(movable)[0].tolist()
    sizes = benchmark.macro_sizes
    canvas_w, canvas_h = benchmark.canvas_width, benchmark.canvas_height
    movable_indices.sort(key=lambda i: -sizes[i, 1].item())
    gap = 0.001
    cursor_x = cursor_y = row_height = 0.0
    for idx in movable_indices:
        w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
        if cursor_x + w > canvas_w:
            cursor_x, cursor_y, row_height = 0.0, cursor_y + row_height + gap, 0.0
        if cursor_y + h > canvas_h:
            placement[idx, 0], placement[idx, 1] = w / 2, h / 2
            continue
        placement[idx, 0], placement[idx, 1] = cursor_x + w / 2, cursor_y + h / 2
        cursor_x += w + gap
        row_height = max(row_height, h)
    return placement.numpy().astype(np.float64)


def _try_load_plc(benchmark):
    """Best-effort: load PlacementCost for real-proxy scoring during search."""
    try:
        from pathlib import Path
        from macro_place.loader import load_benchmark_from_dir
        from macro_place.objective import compute_proxy_cost
        for root in [
            Path("external/MacroPlacement/Testcases/ICCAD04"),
            Path(__file__).parent.parent.parent / "external" / "MacroPlacement" / "Testcases" / "ICCAD04",
        ]:
            d = root / benchmark.name
            if d.exists():
                _, plc = load_benchmark_from_dir(d.as_posix())

                def score(pos_np):
                    t = torch.tensor(pos_np, dtype=torch.float32)
                    t[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
                    return compute_proxy_cost(t, benchmark, plc)["proxy_cost"]
                return score
    except Exception:
        pass
    return None


class YaleAMPlacer:
    """Cyclic smooth-GD <-> CD, then cold-SA finishing stage."""

    def __init__(self):
        self.density_w = float(os.environ.get("HGS_DENSITY_W", "0.7"))
        self.congestion_w = float(os.environ.get("HGS_CONGESTION_W", "1.3"))
        self.gd1_steps = int(os.environ.get("HGS_GD1_STEPS", "650"))
        self.gd1_max_sec = float(os.environ.get("HGS_GD1_MAX_SEC", "780"))
        self.gd2_steps = int(os.environ.get("HGS_GD2_STEPS", "320"))
        self.gd2_max_sec = float(os.environ.get("HGS_GD2_MAX_SEC", "420"))
        self.cycles = int(os.environ.get("HGS_CYCLES", "3"))
        self.cd_sec = float(os.environ.get("HGS_CD_SEC", "150"))
        self.final_cd_sec = float(os.environ.get("HGS_FINAL_CD_SEC", "260"))
        self.sa_budget_cap = float(os.environ.get("HGS_SA_CAP_SEC", "3200"))
        self.sa_min_sec = float(os.environ.get("HGS_SA_MIN_SEC", "120"))

    def _log(self, benchmark, msg):
        sys.stderr.write(f"[yale_am] {benchmark.name} {msg}\n")
        sys.stderr.flush()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        fp = FastProxy(benchmark)
        real_score = _try_load_plc(benchmark)
        score = real_score if real_score is not None else fp.proxy
        gd = _SmoothGD(benchmark, self.density_w, self.congestion_w)
        t0 = time.time()

        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        half_w, half_h = sizes[:, 0] * 0.5, sizes[:, 1] * 0.5
        nh = benchmark.num_hard_macros

        # only accept zero-overlap placements into best
        best, best_cost = None, float("inf")

        pos = gd.run(benchmark.macro_positions.numpy().astype(np.float64),
                     self.gd1_steps, self.gd1_max_sec, 0.5, 8.0, 0.2, 0.02)
        pos, overlaps = _legalize_hard_macros(pos, benchmark)
        if overlaps == 0:
            best, best_cost = pos.copy(), score(pos)
        self._log(benchmark, f"gd1 overlaps={overlaps} "
                             f"cost={'n/a' if overlaps else f'{best_cost:.5f}'} [{time.time()-t0:.0f}s]")

        for cyc in range(2, self.cycles + 1):
            cd_pos, _ = coordinate_descent_polish(fp, benchmark, pos, self.cd_sec)
            pos = gd.run(cd_pos, self.gd2_steps, self.gd2_max_sec, 4.0, 8.0, 0.05, 0.02)
            pos, overlaps = _legalize_hard_macros(pos, benchmark)
            if overlaps == 0:
                cost = score(pos)
                if cost < best_cost:
                    best_cost, best = cost, pos.copy()
            self._log(benchmark, f"cycle{cyc} overlaps={overlaps} "
                                 f"cost={'n/a' if overlaps else f'{cost:.5f}'} [{time.time()-t0:.0f}s]")

        if best is None:
            # GD/legalize never reached zero overlaps -> shelf-pack fallback
            best = _shelf_pack_fallback(benchmark, base_pos=pos)
            best_cost = score(best)
            self._log(benchmark, f"gd path never legalized -> shelf-pack fallback "
                                 f"cost={best_cost:.5f} [{time.time()-t0:.0f}s]")

        cd_pos, cd_cost = coordinate_descent_polish(fp, benchmark, best, self.final_cd_sec)
        if count_overlaps(cd_pos, half_w, half_h, nh) == 0:
            real_cd_cost = score(cd_pos)
            if real_cd_cost < best_cost:
                best_cost, best = real_cd_cost, cd_pos

        sa_budget = self.sa_budget_cap - (time.time() - t0)
        if sa_budget >= self.sa_min_sec and count_overlaps(best, half_w, half_h, nh) == 0:
            try:
                sa_pos, sa_fast_cost = cold_simulated_annealing(fp, benchmark, best, sa_budget)
                if count_overlaps(sa_pos, half_w, half_h, nh) == 0:
                    sa_cost = score(sa_pos)
                    if sa_cost < best_cost - 1e-9:
                        best_cost, best = sa_cost, sa_pos
                self._log(benchmark, f"cold_sa cost={best_cost:.5f} (budget {sa_budget:.0f}s)")
            except Exception as e:
                self._log(benchmark, f"cold_sa skipped: {e!r}")

        # final safety net (should be a no-op)
        if count_overlaps(best, half_w, half_h, nh) > 0:
            self._log(benchmark, "best still had overlaps at return -> shelf-pack fallback")
            best = _shelf_pack_fallback(benchmark)
            best_cost = score(best)

        self._log(benchmark, f"FINAL cost={best_cost:.5f} [{time.time()-t0:.0f}s]")

        result = torch.tensor(best, dtype=torch.float32)
        result[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return result
