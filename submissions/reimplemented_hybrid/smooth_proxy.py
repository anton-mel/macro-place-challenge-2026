"""
smooth_proxy.py — differentiable relaxation of the TILOS proxy cost.
yale_am (Anton Melnychuk).

wirelength: soft bbox via log-sum-exp
density: exact overlap-area grid, power-mean top-K surrogate
congestion: sigmoid-softened L/T-route demand grid, median-row 3-pin nets

Used only to steer gradient descent; final placement scored by the
unmodified official evaluator.
"""
from __future__ import annotations

import torch


def soft_bbox_wirelength(pin_xy, owner, offset, pin_mask, net_weight,
                          net_count, canvas_w, canvas_h, gamma):
    """LSE-relaxed half-perimeter wirelength."""
    pins = pin_xy[owner] + offset  # [n_nets, max_pins, 2]
    neg_fill, pos_fill = -1e20, 1e20
    x = torch.where(pin_mask, pins[..., 0], pins[..., 0].new_full((), neg_fill))
    x_neg = torch.where(pin_mask, pins[..., 0], pins[..., 0].new_full((), pos_fill))
    y = torch.where(pin_mask, pins[..., 1], pins[..., 1].new_full((), neg_fill))
    y_neg = torch.where(pin_mask, pins[..., 1], pins[..., 1].new_full((), pos_fill))

    x_max = torch.logsumexp(gamma * x, dim=1) / gamma
    x_min = -torch.logsumexp(-gamma * x_neg, dim=1) / gamma
    y_max = torch.logsumexp(gamma * y, dim=1) / gamma
    y_min = -torch.logsumexp(-gamma * y_neg, dim=1) / gamma

    half_perimeter = (x_max - x_min) + (y_max - y_min)
    return (half_perimeter * net_weight).sum() / (net_count * (canvas_w + canvas_h))


def overlap_area_grid(macro_xy, macro_wh, grid_cols, grid_rows, canvas_w, canvas_h):
    """Exact overlap area between every macro and every grid cell, [rows, cols]."""
    cell_w, cell_h = canvas_w / grid_cols, canvas_h / grid_rows
    half_w = macro_wh[:, 0:1] * 0.5
    half_h = macro_wh[:, 1:2] * 0.5
    col_centers = (torch.arange(grid_cols, device=macro_xy.device, dtype=macro_xy.dtype) + 0.5) * cell_w
    row_centers = (torch.arange(grid_rows, device=macro_xy.device, dtype=macro_xy.dtype) + 0.5) * cell_h

    x_overlap = torch.relu(
        torch.minimum(macro_xy[:, 0:1] + half_w, col_centers[None, :] + cell_w / 2)
        - torch.maximum(macro_xy[:, 0:1] - half_w, col_centers[None, :] - cell_w / 2)
    )
    y_overlap = torch.relu(
        torch.minimum(macro_xy[:, 1:2] + half_h, row_centers[None, :] + cell_h / 2)
        - torch.maximum(macro_xy[:, 1:2] - half_h, row_centers[None, :] - cell_h / 2)
    )
    return torch.einsum("mr,mc->rc", y_overlap, x_overlap) / (cell_w * cell_h)


def power_mean(values, p):
    """Smooth top-K surrogate, sharpens toward max() as p grows."""
    return values.flatten().clamp(min=1e-12).pow(p).mean().pow(1.0 / p)


def routing_demand(pin_xy, owner, offset, pin_mask, net_weight,
                    grid_cols, grid_rows, canvas_w, canvas_h,
                    h_tracks_per_micron, v_tracks_per_micron, band_softness=0.5):
    """Soft L/T-route demand grids (vertical_demand, horizontal_demand).

    Pin 0 = driver, rest = sinks. 3-pin nets route as one span at their
    median row (T-route) instead of a 2-pin star (L-route).
    """
    pins = pin_xy[owner] + offset
    cell_w, cell_h = canvas_w / grid_cols, canvas_h / grid_rows
    softness = max(cell_w, cell_h) * band_softness
    col_centers = (torch.arange(grid_cols, device=pins.device, dtype=pins.dtype) + 0.5) * cell_w
    row_centers = (torch.arange(grid_rows, device=pins.device, dtype=pins.dtype) + 0.5) * cell_h
    mask_f = pin_mask.to(pins.dtype)

    driver_x, driver_y = pins[:, 0, 0], pins[:, 0, 1]
    big = float(canvas_h * 100.0)
    y_hi = torch.where(pin_mask, pins[..., 1], pins[..., 1].new_full((), -big))
    y_lo = torch.where(pin_mask, pins[..., 1], pins[..., 1].new_full((), big))
    live_pins = mask_f.sum(dim=1)
    is_three_pin = ((live_pins > 2.5) & (live_pins < 3.5)).to(pins.dtype)
    median_y = (pins[..., 1] * mask_f).sum(dim=1) - y_hi.max(dim=1).values - y_lo.min(dim=1).values
    driver_row_y = is_three_pin * median_y + (1.0 - is_three_pin) * driver_y

    row_hit = (torch.sigmoid((row_centers[None, :] - (driver_row_y[:, None] - cell_h / 2)) / softness)
               * torch.sigmoid(((driver_row_y[:, None] + cell_h / 2) - row_centers[None, :]) / softness))

    sink_x, sink_y = pins[:, 1:, 0], pins[:, 1:, 1]
    sink_mask = pin_mask[:, 1:].to(pins.dtype)
    x_lo = torch.minimum(driver_x[:, None], sink_x)
    x_hi = torch.maximum(driver_x[:, None], sink_x)
    y_lo_pair = torch.minimum(driver_y[:, None], sink_y)
    y_hi_pair = torch.maximum(driver_y[:, None], sink_y)

    # horizontal demand: star (per-sink column span) or 3-pin single span
    col_in_span_pair = (torch.sigmoid((col_centers[None, None, :] - x_lo[:, :, None]) / softness)
                         * torch.sigmoid((x_hi[:, :, None] - col_centers[None, None, :]) / softness))
    h_star = (col_in_span_pair * sink_mask[:, :, None]).sum(dim=1) * net_weight[:, None]
    x_hi_net = torch.where(pin_mask, pins[..., 0], pins[..., 0].new_full((), -big)).max(dim=1).values
    x_lo_net = torch.where(pin_mask, pins[..., 0], pins[..., 0].new_full((), big)).min(dim=1).values
    col_in_net_span = (torch.sigmoid((col_centers[None, :] - x_lo_net[:, None]) / softness)
                        * torch.sigmoid((x_hi_net[:, None] - col_centers[None, :]) / softness))
    h_tpin = col_in_net_span * net_weight[:, None]
    h_cols = is_three_pin[:, None] * h_tpin + (1.0 - is_three_pin)[:, None] * h_star
    horizontal_demand = torch.einsum("nr,nc->rc", row_hit, h_cols)

    # vertical demand: each sink's column band over the pair's row span
    sink_col_hit = (torch.sigmoid((col_centers[None, None, :] - (sink_x[:, :, None] - cell_w / 2)) / softness)
                     * torch.sigmoid(((sink_x[:, :, None] + cell_w / 2) - col_centers[None, None, :]) / softness))
    row_in_span_pair = (torch.sigmoid((row_centers[None, None, :] - y_lo_pair[:, :, None]) / softness)
                         * torch.sigmoid((y_hi_pair[:, :, None] - row_centers[None, None, :]) / softness))
    weighted_rows = row_in_span_pair * sink_mask[:, :, None] * net_weight[:, None, None]
    vertical_demand = torch.einsum("nkr,nkc->rc", weighted_rows, sink_col_hit)

    return (vertical_demand / max(cell_w * v_tracks_per_micron, 1e-9),
            horizontal_demand / max(cell_h * h_tracks_per_micron, 1e-9))


def macro_blockage(hard_xy, hard_wh, grid_cols, grid_rows, canvas_w, canvas_h,
                    h_alloc, v_alloc, h_tracks_per_micron, v_tracks_per_micron):
    """Routing capacity consumed by a hard macro physically covering a row/column."""
    cell_w, cell_h = canvas_w / grid_cols, canvas_h / grid_rows
    half_w, half_h = hard_wh[:, 0:1] * 0.5, hard_wh[:, 1:2] * 0.5
    col_centers = (torch.arange(grid_cols, device=hard_xy.device, dtype=hard_xy.dtype) + 0.5) * cell_w
    row_centers = (torch.arange(grid_rows, device=hard_xy.device, dtype=hard_xy.dtype) + 0.5) * cell_h
    x_cover = torch.relu(torch.minimum(hard_xy[:, 0:1] + half_w, col_centers[None, :] + cell_w / 2)
                          - torch.maximum(hard_xy[:, 0:1] - half_w, col_centers[None, :] - cell_w / 2))
    y_cover = torch.relu(torch.minimum(hard_xy[:, 1:2] + half_h, row_centers[None, :] + cell_h / 2)
                          - torch.maximum(hard_xy[:, 1:2] - half_h, row_centers[None, :] - cell_h / 2))
    v_block = torch.einsum("mr,mc->rc", y_cover / cell_h, x_cover) * v_alloc
    h_block = torch.einsum("mr,mc->rc", y_cover, x_cover / cell_w) * h_alloc
    return (v_block / max(cell_w * v_tracks_per_micron, 1e-9),
            h_block / max(cell_h * h_tracks_per_micron, 1e-9))


def smooth_costs(pin_xy, macro_wh, owner, offset, pin_mask, net_weight, net_count,
                  grid_cols, grid_rows, canvas_w, canvas_h,
                  h_tracks_per_micron, v_tracks_per_micron, h_alloc, v_alloc,
                  n_hard, n_macros, gamma_wl=6.0, p_density=10.0, p_congestion=16.0):
    """Returns (wirelength, 0.5*density, 0.5*congestion); sum == smooth proxy."""
    wl = soft_bbox_wirelength(pin_xy, owner, offset, pin_mask, net_weight,
                               net_count, canvas_w, canvas_h, gamma_wl)

    density_grid = overlap_area_grid(pin_xy[:n_macros], macro_wh[:n_macros],
                                      grid_cols, grid_rows, canvas_w, canvas_h)
    density_cost = 0.5 * power_mean(density_grid, p_density)

    v_net, h_net = routing_demand(pin_xy, owner, offset, pin_mask, net_weight,
                                   grid_cols, grid_rows, canvas_w, canvas_h,
                                   h_tracks_per_micron, v_tracks_per_micron)
    v_block, h_block = macro_blockage(pin_xy[:n_hard], macro_wh[:n_hard],
                                       grid_cols, grid_rows, canvas_w, canvas_h,
                                       h_alloc, v_alloc, h_tracks_per_micron, v_tracks_per_micron)
    congestion_cost = 0.5 * power_mean(
        torch.cat([(v_net + v_block).flatten(), (h_net + h_block).flatten()]), p_congestion)

    return wl, density_cost, congestion_cost
