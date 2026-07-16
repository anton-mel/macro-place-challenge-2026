"""
local_search.py — discrete refinement after smooth-GD, scored via fast_proxy.
yale_am (Anton Melnychuk).

coordinate_descent_polish: per-macro candidate sampling + same-size swaps.
cold_simulated_annealing: low-temp Metropolis, includes soft macros.
"""
import math
import random
import time

import numpy as np

from legalize import count_overlaps, overlaps_if_moved_to


def coordinate_descent_polish(fp, benchmark, base_pos, time_budget,
                               candidates_per_macro=10, seed=12345):
    nh = benchmark.num_hard_macros
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] * 0.5, sizes[:, 1] * 0.5
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    movable = (benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()).numpy()
    movable_idx = np.where(movable)[0]
    if len(movable_idx) == 0:
        return base_pos, fp.proxy(base_pos)

    cur = base_pos.astype(np.float64).copy()
    cur_cost = fp.proxy(cur)
    best, best_cost = cur.copy(), cur_cost
    diag = math.hypot(cw, ch)
    scales = np.array([0.003, 0.01, 0.03, 0.08, 0.16]) * diag
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed + 1)
    t0 = time.time()

    while time.time() - t0 < time_budget:
        order = movable_idx.tolist()
        py_rng.shuffle(order)
        improved = False

        for idx in order:
            if time.time() - t0 > time_budget:
                break
            ox, oy = cur[idx, 0], cur[idx, 1]
            best_local_cost, bx, by = cur_cost, ox, oy
            for _ in range(candidates_per_macro):
                scale = scales[np_rng.integers(len(scales))]
                nx = np.clip(ox + np_rng.standard_normal() * scale, half_w[idx], cw - half_w[idx])
                ny = np.clip(oy + np_rng.standard_normal() * scale, half_h[idx], ch - half_h[idx])
                if overlaps_if_moved_to(cur, half_w, half_h, idx, nh, nx, ny):
                    continue
                cur[idx, 0], cur[idx, 1] = nx, ny
                cost = fp.proxy(cur)
                if cost < best_local_cost - 1e-9:
                    best_local_cost, bx, by = cost, nx, ny
            cur[idx, 0], cur[idx, 1] = bx, by
            if best_local_cost < cur_cost - 1e-9:
                cur_cost = best_local_cost
                improved = True

        # same-size macro swaps
        n_swaps = max(300, len(movable_idx) * 5)
        for _ in range(n_swaps):
            if time.time() - t0 > time_budget:
                break
            i = int(movable_idx[np_rng.integers(len(movable_idx))])
            j = int(movable_idx[np_rng.integers(len(movable_idx))])
            if i == j:
                continue
            area_i, area_j = sizes[i, 0] * sizes[i, 1], sizes[j, 0] * sizes[j, 1]
            if area_j < 0.5 * area_i or area_j > 2.0 * area_i:
                continue
            oxi, oyi, oxj, oyj = cur[i, 0], cur[i, 1], cur[j, 0], cur[j, 1]
            nxi = np.clip(oxj, half_w[i], cw - half_w[i])
            nyi = np.clip(oyj, half_h[i], ch - half_h[i])
            nxj = np.clip(oxi, half_w[j], cw - half_w[j])
            nyj = np.clip(oyi, half_h[j], ch - half_h[j])
            cur[i, 0], cur[i, 1] = nxi, nyi
            cur[j, 0], cur[j, 1] = nxj, nyj
            if (overlaps_if_moved_to(cur, half_w, half_h, i, nh, nxi, nyi)
                    or overlaps_if_moved_to(cur, half_w, half_h, j, nh, nxj, nyj)):
                cur[i, 0], cur[i, 1], cur[j, 0], cur[j, 1] = oxi, oyi, oxj, oyj
                continue
            cost = fp.proxy(cur)
            if cost < cur_cost - 1e-9:
                cur_cost, improved = cost, True
            else:
                cur[i, 0], cur[i, 1], cur[j, 0], cur[j, 1] = oxi, oyi, oxj, oyj

        if cur_cost < best_cost - 1e-12:
            best_cost, best = cur_cost, cur.copy()
        if not improved:
            break

    if count_overlaps(best, half_w, half_h, nh) > 0:
        return base_pos, fp.proxy(base_pos)
    return best, best_cost


def cold_simulated_annealing(fp, benchmark, base_pos, time_budget,
                              t_init=1e-6, t_end=1e-7, move_min=0.1, move_max=5.0,
                              swap_prob=0.10, soft_prob=0.50, seed=0, max_iters=200_000_000):
    """Near-greedy Metropolis (T~1e-6): single-macro moves, ~half on soft
    macros, plus same-area hard-macro swaps."""
    nh, nm = benchmark.num_hard_macros, benchmark.num_macros
    sizes = benchmark.macro_sizes.numpy().astype(np.float64)
    half_w, half_h = sizes[:, 0] * 0.5, sizes[:, 1] * 0.5
    cw, ch = benchmark.canvas_width, benchmark.canvas_height
    fixed = benchmark.macro_fixed.numpy()
    rng = np.random.default_rng(seed)

    areas = np.round(sizes[:nh, 0] * sizes[:nh, 1], 3)
    groups = {}
    for i in range(nh):
        if not fixed[i]:
            groups.setdefault(areas[i], []).append(i)
    swap_groups = [np.array(g, dtype=np.int64) for g in groups.values() if len(g) >= 2]
    can_swap = len(swap_groups) > 0
    n_soft = nm - nh

    cur = base_pos.astype(np.float64).copy()
    cur_cost = fp.proxy(cur)
    best, best_cost = cur.copy(), cur_cost
    cooling_rate = (t_end / t_init) ** (1.0 / max(max_iters - 1, 1))
    T = t_init
    t0 = time.time()
    it = 0

    while it < max_iters:
        if (it & 0xFF) == 0 and time.time() - t0 >= time_budget:
            break
        move_kind = 0  # 0=single macro, 1=swap
        if can_swap and rng.random() < swap_prob:
            grp = swap_groups[rng.integers(len(swap_groups))]
            i, j = int(grp[rng.integers(grp.shape[0])]), int(grp[rng.integers(grp.shape[0])])
            if i == j:
                it += 1; T *= cooling_rate; continue
            oix, oiy, ojx, ojy = cur[i, 0], cur[i, 1], cur[j, 0], cur[j, 1]
            cur[i, 0], cur[i, 1], cur[j, 0], cur[j, 1] = ojx, ojy, oix, oiy
            if overlaps_if_moved_to(cur, half_w, half_h, i, nh, ojx, ojy) or \
               overlaps_if_moved_to(cur, half_w, half_h, j, nh, oix, oiy):
                cur[i, 0], cur[i, 1], cur[j, 0], cur[j, 1] = oix, oiy, ojx, ojy
                it += 1; T *= cooling_rate; continue
            move_kind = 1
        else:
            move_soft = n_soft > 0 and rng.random() < soft_prob
            i = int(nh + rng.integers(n_soft)) if move_soft else int(rng.integers(nh))
            if fixed[i]:
                it += 1; T *= cooling_rate; continue
            angle = rng.random() * 2.0 * math.pi
            magnitude = move_min + rng.random() * (move_max - move_min)
            nx = cur[i, 0] + math.cos(angle) * magnitude
            ny = cur[i, 1] + math.sin(angle) * magnitude
            if nx < half_w[i] or nx > cw - half_w[i] or ny < half_h[i] or ny > ch - half_h[i]:
                it += 1; T *= cooling_rate; continue
            if not move_soft and overlaps_if_moved_to(cur, half_w, half_h, i, nh, nx, ny):
                it += 1; T *= cooling_rate; continue
            oix, oiy = cur[i, 0], cur[i, 1]
            cur[i, 0], cur[i, 1] = nx, ny

        cost = fp.proxy(cur)
        delta = cost - cur_cost
        if delta <= 0.0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            cur_cost = cost
            if cur_cost < best_cost:
                best_cost, best = cur_cost, cur.copy()
        else:
            if move_kind == 1:
                cur[i, 0], cur[i, 1], cur[j, 0], cur[j, 1] = oix, oiy, ojx, ojy
            else:
                cur[i, 0], cur[i, 1] = oix, oiy
        it += 1
        T *= cooling_rate

    if count_overlaps(best, half_w, half_h, nh) > 0:
        return base_pos, fp.proxy(base_pos)
    return best, best_cost
