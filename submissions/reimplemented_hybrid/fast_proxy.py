"""
fast_proxy.py — Numba-jitted proxy evaluator for local search.
yale_am (Anton Melnychuk).

Precomputes flat tables once per benchmark; cheap repeated proxy() calls
for coordinate descent / simulated annealing. Ranking-only, not bit-exact
with the official evaluator, which scores the final placement.
"""
import numpy as np
from numba import njit


def build_pin_tables(benchmark):
    """Flatten benchmark.net_pin_nodes into (owner, offset) arrays."""
    owners, off_x, off_y, starts, counts = [], [], [], [], []
    cursor = 0
    n_hard = benchmark.num_hard_macros
    for net in benchmark.net_pin_nodes:
        if net.shape[0] < 2:
            continue
        starts.append(cursor)
        for row in net.tolist():
            owner, pin_idx = int(row[0]), int(row[1])
            owners.append(owner)
            if owner < n_hard and pin_idx < benchmark.macro_pin_offsets[owner].shape[0]:
                ox, oy = benchmark.macro_pin_offsets[owner][pin_idx].tolist()
            else:
                ox, oy = 0.0, 0.0
            off_x.append(ox)
            off_y.append(oy)
        counts.append(net.shape[0])
        cursor += net.shape[0]
    return dict(
        owner=np.asarray(owners, dtype=np.int64),
        off_x=np.asarray(off_x, dtype=np.float64),
        off_y=np.asarray(off_y, dtype=np.float64),
        net_start=np.asarray(starts, dtype=np.int64),
        net_count=np.asarray(counts, dtype=np.int64),
        net_weight=np.ones(len(starts), dtype=np.float64),
    )


@njit(cache=True, fastmath=False)
def _hpwl(pos_aug, owner, off_x, off_y, net_start, net_count, net_weight, denom):
    n_nets = net_start.shape[0]
    total = 0.0
    for i in range(n_nets):
        s = net_start[i]
        n = net_count[i]
        o = owner[s]
        x = pos_aug[o, 0] + off_x[s]
        y = pos_aug[o, 1] + off_y[s]
        xmin = xmax = x
        ymin = ymax = y
        for p in range(1, n):
            idx = s + p
            o = owner[idx]
            x = pos_aug[o, 0] + off_x[idx]
            y = pos_aug[o, 1] + off_y[idx]
            if x < xmin:
                xmin = x
            elif x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            elif y > ymax:
                ymax = y
        total += net_weight[i] * ((xmax - xmin) + (ymax - ymin))
    return total / denom


@njit(cache=True, fastmath=False)
def _density_grid_add(grid, mx, my, mw, mh, n_macros, n_rows, n_cols, cell_w, cell_h):
    for m in range(n_macros):
        x_min, x_max = mx[m] - mw[m] * 0.5, mx[m] + mw[m] * 0.5
        y_min, y_max = my[m] - mh[m] * 0.5, my[m] + mh[m] * 0.5
        c_lo = max(0, int(x_min // cell_w))
        c_hi = min(n_cols - 1, int(x_max // cell_w))
        r_lo = max(0, int(y_min // cell_h))
        r_hi = min(n_rows - 1, int(y_max // cell_h))
        for r in range(r_lo, r_hi + 1):
            for c in range(c_lo, c_hi + 1):
                xd = min(x_max, (c + 1) * cell_w) - max(x_min, c * cell_w)
                yd = min(y_max, (r + 1) * cell_h) - max(y_min, r * cell_h)
                if xd > 0.0 and yd > 0.0:
                    grid[r * n_cols + c] += xd * yd


@njit(cache=True, fastmath=False)
def _top_k_mean(grid, n_cells, cell_area, top_fraction):
    values = np.empty(n_cells, dtype=np.float64)
    n_occupied = 0
    for k in range(n_cells):
        v = grid[k] / cell_area
        if v != 0.0:
            values[n_occupied] = v
            n_occupied += 1
    if n_occupied == 0:
        return 0.0
    occ = np.sort(values[:n_occupied])[::-1]
    k = max(1, int(n_cells * top_fraction))
    k = min(k, n_occupied)
    total = 0.0
    for i in range(k):
        total += occ[i]
    return total / k


@njit(cache=True, fastmath=False)
def _routing_demand_add(v_grid, h_grid, owner, off_x, off_y, net_start, net_count,
                         net_weight, pos_aug, n_rows, n_cols, cell_w, cell_h):
    """L-route star: driver row gets H demand across sink column span; each
    sink's column gets V demand across its row span."""
    n_nets = net_start.shape[0]
    for i in range(n_nets):
        s = net_start[i]
        n = net_count[i]
        w = net_weight[i]
        do = owner[s]
        dx = pos_aug[do, 0] + off_x[s]
        dy = pos_aug[do, 1] + off_y[s]
        d_row = min(n_rows - 1, max(0, int(dy // cell_h)))
        for p in range(1, n):
            idx = s + p
            so = owner[idx]
            sx = pos_aug[so, 0] + off_x[idx]
            sy = pos_aug[so, 1] + off_y[idx]
            s_col = min(n_cols - 1, max(0, int(sx // cell_w)))
            c_lo, c_hi = (int(dx // cell_w), s_col) if dx <= sx else (s_col, int(dx // cell_w))
            c_lo = max(0, min(n_cols - 1, c_lo))
            c_hi = max(0, min(n_cols - 1, c_hi))
            for c in range(c_lo, c_hi + 1):
                h_grid[d_row * n_cols + c] += w
            r_lo, r_hi = (int(dy // cell_h), int(sy // cell_h)) if dy <= sy else (int(sy // cell_h), int(dy // cell_h))
            r_lo = max(0, min(n_rows - 1, r_lo))
            r_hi = max(0, min(n_rows - 1, r_hi))
            for r in range(r_lo, r_hi + 1):
                v_grid[r * n_cols + s_col] += w


class FastProxy:
    """proxy = WL + 0.5*density + 0.5*congestion, Numba-accelerated."""

    def __init__(self, benchmark):
        self.b = benchmark
        tables = build_pin_tables(benchmark)
        self.owner = tables["owner"]
        self.off_x = tables["off_x"]
        self.off_y = tables["off_y"]
        self.net_start = tables["net_start"]
        self.net_count = tables["net_count"]
        self.net_weight = tables["net_weight"]
        self.wl_denom = max(len(self.net_start), 1) * (benchmark.canvas_width + benchmark.canvas_height)

        self.n_rows = int(benchmark.grid_rows)
        self.n_cols = int(benchmark.grid_cols)
        self.cell_w = benchmark.canvas_width / self.n_cols
        self.cell_h = benchmark.canvas_height / self.n_rows
        self.cell_area = self.cell_w * self.cell_h
        self.n_cells = self.n_rows * self.n_cols

        sizes = benchmark.macro_sizes.numpy().astype(np.float64)
        self.all_w = np.ascontiguousarray(sizes[:, 0])
        self.all_h = np.ascontiguousarray(sizes[:, 1])
        self.n_macros = benchmark.num_macros
        self.h_tracks = benchmark.hroutes_per_micron
        self.v_tracks = benchmark.vroutes_per_micron

    def _pos_aug(self, pos_np):
        ports = self.b.port_positions.numpy().astype(np.float64) if self.b.port_positions.shape[0] else np.zeros((0, 2))
        return np.concatenate([pos_np, ports], axis=0)

    def wirelength(self, pos_np):
        return _hpwl(self._pos_aug(pos_np), self.owner, self.off_x, self.off_y,
                     self.net_start, self.net_count, self.net_weight, self.wl_denom)

    def density(self, pos_np):
        grid = np.zeros(self.n_cells, dtype=np.float64)
        _density_grid_add(grid, np.ascontiguousarray(pos_np[:, 0]), np.ascontiguousarray(pos_np[:, 1]),
                          self.all_w, self.all_h, self.n_macros, self.n_rows, self.n_cols,
                          self.cell_w, self.cell_h)
        return 0.5 * _top_k_mean(grid, self.n_cells, self.cell_area, 0.10)

    def congestion(self, pos_np):
        v_grid = np.zeros(self.n_cells, dtype=np.float64)
        h_grid = np.zeros(self.n_cells, dtype=np.float64)
        pos_aug = self._pos_aug(pos_np)
        _routing_demand_add(v_grid, h_grid, self.owner, self.off_x, self.off_y,
                            self.net_start, self.net_count, self.net_weight, pos_aug,
                            self.n_rows, self.n_cols, self.cell_w, self.cell_h)
        v_grid /= max(self.cell_w * self.v_tracks, 1e-9)
        h_grid /= max(self.cell_h * self.h_tracks, 1e-9)
        combined = np.concatenate([v_grid, h_grid])
        return 0.5 * _top_k_mean(combined, combined.shape[0], 1.0, 0.05)

    def proxy(self, pos_np):
        return self.wirelength(pos_np) + self.density(pos_np) + self.congestion(pos_np)
