"""
Pure-numpy helpers that fit per-node data onto a character-cell pixel grid of arbitrary size.

The node map shows one pixel per node when there is room and otherwise bins runs of contiguous
node ids (rack-aligned when possible) into one pixel. The grid shape depends only on the terminal
size, never on the CDU/rack shape of the machine config.
"""
from dataclasses import dataclass

import numpy as np

# Node state codes used in UISnapshot.node_state
FREE, BUSY, DOWN = 0, 1, 2


@dataclass(frozen=True)
class NodeMapPlan:
    per_bin: int          # nodes folded into one pixel
    n_bins: int           # pixels that carry data
    rack_aligned: bool    # whether pixels are tiled rack by rack
    tile_w: int = 0       # rack tile size in pixels (rack_aligned only)
    tile_h: int = 0
    gap: int = 0          # pixels between rack tiles
    tiles_x: int = 0      # rack tiles per tile row


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def tile_shape(n_px):
    """Roughly square (w, h) with w * h >= n_px."""
    w = int(np.ceil(np.sqrt(n_px)))
    h = int(np.ceil(n_px / w))
    return w, h


def plan_nodemap(total_nodes, nodes_per_rack, width_px, height_px, per_bin_override=None):
    """
    Choose how to fold `total_nodes` onto a `width_px` x `height_px` pixel grid.

    Racks (runs of `nodes_per_rack` contiguous ids) are tiled left to right and wrap to the next
    tile row, so the number of CDUs or racks in the config never matters. Bins are divisors of the
    rack size so a pixel never straddles two racks. If even one pixel per rack does not fit, the
    nodes are binned in plain row-major order instead.
    """
    width_px, height_px = max(width_px, 1), max(height_px, 1)
    if nodes_per_rack and nodes_per_rack > 1 and total_nodes % nodes_per_rack == 0:
        n_racks = total_nodes // nodes_per_rack
        candidates = [per_bin_override] if per_bin_override in _divisors(nodes_per_rack) else _divisors(nodes_per_rack)
        for per_bin in candidates:
            px_per_rack = nodes_per_rack // per_bin
            tw, th = tile_shape(px_per_rack)
            for gap in (1, 0):
                tiles_x = (width_px + gap) // (tw + gap)
                tiles_y = (height_px + gap) // (th + gap)
                if tiles_x * tiles_y >= n_racks and tiles_x > 0:
                    # Only use gaps when they fit; otherwise fall through to gap 0
                    return NodeMapPlan(per_bin, n_racks * px_per_rack, True, tw, th, gap, tiles_x)
    per_bin = per_bin_override or max(1, int(np.ceil(total_nodes / (width_px * height_px))))
    return NodeMapPlan(per_bin, int(np.ceil(total_nodes / per_bin)), False)


def pixel_bin_index(plan, total_nodes, nodes_per_rack, width_px, height_px):
    """
    int32 array [height_px, width_px] mapping each pixel to a bin index (-1 for no data).
    Bin b covers node ids [b * per_bin, (b + 1) * per_bin).
    """
    idx = np.full((height_px, width_px), -1, dtype=np.int32)
    if not plan.rack_aligned:
        flat = np.arange(width_px * height_px, dtype=np.int32)
        flat[flat >= plan.n_bins] = -1
        return flat.reshape(height_px, width_px)
    px_per_rack = nodes_per_rack // plan.per_bin
    n_racks = total_nodes // nodes_per_rack
    tw, th, gap = plan.tile_w, plan.tile_h, plan.gap
    r = np.arange(n_racks)
    ox = (r % plan.tiles_x) * (tw + gap)
    oy = (r // plan.tiles_x) * (th + gap)
    p = np.arange(px_per_rack)
    x = ox[:, None] + (p % tw)[None, :]
    y = oy[:, None] + (p // tw)[None, :]
    b = r[:, None] * px_per_rack + p[None, :]
    ok = (x < width_px) & (y < height_px)
    idx[y[ok], x[ok]] = b[ok]
    return idx


def rack_of_pixels(plan, width_px, height_px, n_racks):
    """int32 array [height_px, width_px]: the rack tile a pixel belongs to (-1 for gaps/empty)."""
    out = np.full((height_px, width_px), -1, dtype=np.int32)
    if not plan.rack_aligned:
        return out
    tw, th, gap = plan.tile_w, plan.tile_h, plan.gap
    r = np.arange(n_racks)
    ox = (r % plan.tiles_x) * (tw + gap)
    oy = (r // plan.tiles_x) * (th + gap)
    for dy in range(th):
        for dx in range(tw):
            x, y = ox + dx, oy + dy
            ok = (x < width_px) & (y < height_px)
            out[y[ok], x[ok]] = r[ok]
    return out


def bin_nodes(node_state, node_power, node_job, per_bin):
    """
    Fold per-node arrays into bins of `per_bin` contiguous nodes.

    Returns a dict of float32/int arrays of length n_bins: `count` (nodes in the bin), the
    free/busy/down fractions (summing to 1 for non-empty bins), `power` (mean node power) and
    `job` (the largest job id among the bin's nodes, -1 if none; a cheap representative).
    """
    total = node_state.shape[0]
    n_bins = -(-total // per_bin)
    pad = n_bins * per_bin - total

    def fold(a, fill):
        if pad:
            a = np.concatenate([a, np.full(pad, fill, dtype=a.dtype)])
        return a.reshape(n_bins, per_bin)

    st = fold(node_state, 255)
    count = (st != 255).sum(axis=1)
    denom = np.maximum(count, 1).astype(np.float32)
    out = {
        "count": count,
        "free": (st == FREE).sum(axis=1) / denom,
        "busy": (st == BUSY).sum(axis=1) / denom,
        "down": (st == DOWN).sum(axis=1) / denom,
        "power": fold(node_power.astype(np.float32), 0.0).sum(axis=1) / denom,
        "job": fold(node_job, -1).max(axis=1),
    }
    return out


# Colors (RGB uint8)
COLOR_FREE = np.array([38, 44, 52], dtype=np.uint8)
COLOR_BUSY = np.array([64, 200, 110], dtype=np.uint8)
COLOR_DOWN = np.array([220, 60, 60], dtype=np.uint8)
COLOR_EMPTY = np.array([0, 0, 0], dtype=np.uint8)

# A compact viridis-like ramp for heatmaps
_RAMP = np.array([
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
], dtype=np.float32)


def heat_colors(values, lo, hi):
    """Map float values to RGB uint8 [..., 3] along the ramp; NaN maps to the empty color."""
    v = np.asarray(values, dtype=np.float32)
    span = hi - lo if hi > lo else 1.0
    t = np.clip((v - lo) / span, 0.0, 1.0)
    pos = np.nan_to_num(t) * (len(_RAMP) - 1)
    i0 = np.floor(pos).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(_RAMP) - 1)
    f = (pos - i0)[..., None]
    rgb = _RAMP[i0] * (1 - f) + _RAMP[i1] * f
    rgb = rgb.astype(np.uint8)
    rgb[np.isnan(v)] = COLOR_EMPTY
    return rgb


def job_colors(job_ids):
    """Hashed palette: a stable, well-spread color per job id (a negative id means no job)."""
    ji = np.asarray(job_ids).astype(np.int64)
    j = ji.astype(np.uint64)
    h = (j * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    hue = (h % np.uint64(360)).astype(np.float32) / 360.0
    # HSV with S=0.65, V=0.95 and a little value jitter to separate neighbouring hues
    v = 0.75 + 0.2 * (((h >> np.uint64(9)) % np.uint64(4)).astype(np.float32) / 3.0)
    s = 0.65
    h6 = hue * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    rgb = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    rgb[ji < 0] = COLOR_FREE
    return rgb


def bin_colors(bins, mode, power_lo=0.0, power_hi=1.0):
    """RGB uint8 [n_bins, 3] for one bin set in the given color mode: state, job or power."""
    n = len(bins["count"])
    if mode == "job":
        rgb = job_colors(bins["job"])
        rgb[bins["down"] > 0] = COLOR_DOWN
        return rgb
    if mode == "power":
        rgb = heat_colors(bins["power"], power_lo, power_hi)
        rgb[bins["down"] >= 0.5] = COLOR_DOWN
        return rgb
    # state: free grey blended towards green by busy fraction, red by down fraction
    busy = bins["busy"][:, None]
    down = bins["down"][:, None]
    # Busy nodes brighten with their power so a running machine visibly pulses
    p = np.clip((bins["power"] - power_lo) / max(power_hi - power_lo, 1e-9), 0, 1)[:, None]
    busy_color = COLOR_BUSY.astype(np.float32) * (0.55 + 0.45 * p)
    rgb = (COLOR_FREE.astype(np.float32) * (1 - busy - down)
           + busy_color * busy + COLOR_DOWN.astype(np.float32) * down)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb = np.where(bins["count"][:, None] > 0, rgb, COLOR_EMPTY)
    assert rgb.shape == (n, 3)
    return rgb


def render_pixels(plan, pix_bin, bin_rgb):
    """RGB uint8 [H, W, 3] image of the bins laid out by `pix_bin`."""
    img = np.zeros(pix_bin.shape + (3,), dtype=np.uint8)
    ok = pix_bin >= 0
    img[ok] = bin_rgb[pix_bin[ok]]
    return img


def plan_blocks(n, width_px, height_px):
    """
    Fit `n` equal items on the grid as square blocks of k x k pixels (1 px gap inside a block when
    k >= 3), as large as possible. If even k = 1 does not fit, items are grouped `group` per pixel.
    Returns (k, group, cols) where cols is the number of blocks per row.
    """
    width_px, height_px = max(width_px, 1), max(height_px, 1)
    if n > width_px * height_px:
        return 1, int(np.ceil(n / (width_px * height_px))), width_px
    for k in range(min(width_px, height_px), 0, -1):
        cols = width_px // k
        if cols * (height_px // k) >= n:
            return k, 1, cols
    return 1, 1, width_px


def block_image(rgb, k, cols, width_px, height_px):
    """RGB image [height_px, width_px, 3] of one color per block, row-major, k x k pixels each."""
    n = len(rgb)
    rows = -(-n // cols)
    grid = np.zeros((rows * cols, 3), dtype=np.uint8)
    grid[:n] = rgb
    big = np.repeat(np.repeat(grid.reshape(rows, cols, 3), k, axis=0), k, axis=1)
    if k >= 3:  # separator line on the right and bottom edge of every block
        big[k - 1::k, :] = 0
        big[:, k - 1::k] = 0
    img = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    h, w = min(height_px, big.shape[0]), min(width_px, big.shape[1])
    img[:h, :w] = big[:h, :w]
    return img


def group_mean(values, group):
    """Mean of consecutive groups of `group` values (last group may be short)."""
    if group <= 1:
        return np.asarray(values)
    n = -(-len(values) // group)
    pad = n * group - len(values)
    v = np.concatenate([values, np.full(pad, np.nan)]) if pad else np.asarray(values)
    return np.nanmean(v.reshape(n, group), axis=1)
