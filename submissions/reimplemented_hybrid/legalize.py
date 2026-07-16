"""
legalize.py — resolve hard-macro overlaps to zero.
yale_am (Anton Melnychuk).

Pairwise push-apart along the smaller-correction axis, with jitter +
re-push escape for stuck dense clusters.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def count_overlaps(pos, half_w, half_h, n_hard):
    count = 0
    for i in range(n_hard):
        for j in range(i + 1, n_hard):
            if (abs(pos[i, 0] - pos[j, 0]) < half_w[i] + half_w[j]
                    and abs(pos[i, 1] - pos[j, 1]) < half_h[i] + half_h[j]):
                count += 1
    return count


@njit(cache=True)
def overlaps_if_moved_to(pos, half_w, half_h, idx, n_hard, new_x, new_y):
    hw_i, hh_i = half_w[idx], half_h[idx]
    for j in range(n_hard):
        if j == idx:
            continue
        if abs(new_x - pos[j, 0]) < hw_i + half_w[j] and abs(new_y - pos[j, 1]) < hh_i + half_h[j]:
            return True
    return False


@njit(cache=True)
def push_apart(pos, half_w, half_h, movable, n_hard, canvas_w, canvas_h, max_passes):
    """One round of pairwise separation; mutates `pos` in place."""
    margin = 0.02
    for _ in range(max_passes):
        found = False
        for i in range(n_hard):
            for j in range(i + 1, n_hard):
                dx = abs(pos[i, 0] - pos[j, 0])
                dy = abs(pos[i, 1] - pos[j, 1])
                need_x = half_w[i] + half_w[j] + margin
                need_y = half_h[i] + half_h[j] + margin
                if dx < need_x and dy < need_y:
                    found = True
                    push_x = need_x - dx
                    push_y = need_y - dy
                    if push_x < push_y:
                        sign = 1.0 if pos[i, 0] < pos[j, 0] else -1.0
                        if movable[i] and movable[j]:
                            pos[i, 0] -= sign * push_x * 0.5
                            pos[j, 0] += sign * push_x * 0.5
                        elif movable[i]:
                            pos[i, 0] -= sign * push_x
                        elif movable[j]:
                            pos[j, 0] += sign * push_x
                    else:
                        sign = 1.0 if pos[i, 1] < pos[j, 1] else -1.0
                        if movable[i] and movable[j]:
                            pos[i, 1] -= sign * push_y * 0.5
                            pos[j, 1] += sign * push_y * 0.5
                        elif movable[i]:
                            pos[i, 1] -= sign * push_y
                        elif movable[j]:
                            pos[j, 1] += sign * push_y
                    for k in (i, j):
                        if movable[k]:
                            if pos[k, 0] < half_w[k]:
                                pos[k, 0] = half_w[k]
                            elif pos[k, 0] > canvas_w - half_w[k]:
                                pos[k, 0] = canvas_w - half_w[k]
                            if pos[k, 1] < half_h[k]:
                                pos[k, 1] = half_h[k]
                            elif pos[k, 1] > canvas_h - half_h[k]:
                                pos[k, 1] = canvas_h - half_h[k]
        if not found:
            return


def legalize(pos, half_w, half_h, movable, n_hard, canvas_w, canvas_h,
             jitter_rounds=60, seed=20260715):
    """Return (legal_positions, remaining_overlap_count)."""
    p = pos.astype(np.float64).copy()
    push_apart(p, half_w, half_h, movable, n_hard, canvas_w, canvas_h, 8000)
    overlaps = count_overlaps(p, half_w, half_h, n_hard)
    if overlaps == 0:
        return p, 0

    diag = (canvas_w ** 2 + canvas_h ** 2) ** 0.5
    rng = np.random.default_rng(seed)
    best_p, best_overlaps = p.copy(), overlaps
    for round_i in range(jitter_rounds):
        amplitude = diag * (0.01 + 0.05 * (round_i / jitter_rounds))
        trial = best_p.copy()
        for i in range(n_hard):
            if movable[i]:
                trial[i, 0] = np.clip(trial[i, 0] + rng.uniform(-amplitude, amplitude),
                                       half_w[i], canvas_w - half_w[i])
                trial[i, 1] = np.clip(trial[i, 1] + rng.uniform(-amplitude, amplitude),
                                       half_h[i], canvas_h - half_h[i])
        push_apart(trial, half_w, half_h, movable, n_hard, canvas_w, canvas_h, 8000)
        trial_overlaps = count_overlaps(trial, half_w, half_h, n_hard)
        if trial_overlaps < best_overlaps:
            best_p, best_overlaps = trial.copy(), trial_overlaps
            if best_overlaps == 0:
                break
    return best_p, best_overlaps
