"""Node Map: one pixel per node (or per bin of contiguous nodes) colored by state, job or power."""
import numpy as np
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable
from raps.ui.widgets.textpane import TextPane

from raps.ui.binning import (
    FREE, BUSY, DOWN, MISSING, plan_nodemap, pixel_bin_index, rack_of_pixels, bin_nodes, bin_colors, render_pixels,
)
from raps.ui.views.base import View, sync_table
from raps.ui.widgets.pixelgrid import PixelGrid

MODES = ["state", "job", "power"]
LEGENDS = {
    "state": "[#262c34]█[/] free  [#40c86e]█[/] busy (brighter = more power)  [#dc3c3c]█[/] down",
    "job": "each color is a job (hashed); [#262c34]█[/] free  [#dc3c3c]█[/] down",
    "power": "node power, low [#440154]█[/][#2a788e]█[/][#22a884]█[/][#7ad151]█[/][#fde725]█[/] high; "
             "[#dc3c3c]█[/] down",
}
HIGHLIGHT = np.array([255, 170, 0], dtype=np.uint8)
STATE_NAMES = {FREE: "free", BUSY: "busy", DOWN: "down", MISSING: "missing"}


class RackTable(DataTable):
    """Rack node table; left/right step to the previous/next rack instead of moving a column cursor."""

    def action_cursor_left(self):
        self.screen.query_one(NodeMap).action_step_rack(-1)

    def action_cursor_right(self):
        self.screen.query_one(NodeMap).action_step_rack(1)


class NodeMap(View):
    view_key = "3"
    title = "Node Map"
    can_focus = True
    BINDINGS = [
        Binding("c", "color_mode", "Color mode", show=False),
        Binding("left", "move(-1, 0)", show=False),
        Binding("right", "move(1, 0)", show=False),
        Binding("up", "move(0, -1)", show=False),
        Binding("down", "move(0, 1)", show=False),
        Binding("enter", "zoom", "Zoom", show=False),
        Binding("escape", "back", "Back", show=False),
    ]
    DEFAULT_CSS = """
    NodeMap #nm-info { height: 3; padding: 0 1; }
    NodeMap #nm-grid { border: round $primary-darken-2; }
    NodeMap #nm-rack { display: none; height: 1fr; }
    NodeMap.zoomed #nm-grid { display: none; }
    NodeMap.zoomed #nm-rack { display: block; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mode = 0
        self.cursor = 0
        self.zoomed = False
        self.focus_job = None   # node-array id of a job to highlight (everything else is dimmed)
        self.focus_label = ""
        self._plan = None
        self._pix = None
        self._rack_px = None
        self._img_size = None

    def compose(self) -> ComposeResult:
        yield TextPane(id="nm-info")
        yield PixelGrid(id="nm-grid")
        yield RackTable(id="nm-rack", cursor_type="row", zebra_stripes=True)

    def on_mount(self):
        self.query_one("#nm-grid", PixelGrid).set_source(self._image)
        self.query_one("#nm-rack", DataTable).add_columns("Node", "State", "Power W", "Job", "Name")

    def on_show(self):
        super().on_show()
        self.focus()

    # -- actions -----------------------------------------------------------------------------
    def action_color_mode(self):
        self.mode = (self.mode + 1) % len(MODES)
        self._redraw()

    def action_step_rack(self, d: int):
        """In the zoomed rack view, switch to the previous/next rack."""
        if self.snapshot is None:
            return
        m = self.snapshot.meta
        n_racks = m.total_nodes // m.nodes_per_rack
        self.cursor = min(max(self.cursor + d, 0), n_racks - 1)
        self._redraw()
        self._fill_rack(self.snapshot)

    def action_move(self, dx: int, dy: int):
        if self.zoomed:
            self.action_step_rack(dx)
            return
        if self._plan is None or not self._plan.rack_aligned or self.snapshot is None:
            return
        m = self.snapshot.meta
        n_racks = m.total_nodes // m.nodes_per_rack
        tx = self._plan.tiles_x
        x = min(max(self.cursor % tx + dx, 0), tx - 1)
        y = max(self.cursor // tx + dy, 0)
        self.cursor = min(y * tx + x, n_racks - 1)
        self._redraw()

    def show_job(self, job_id, rack: int):
        """Highlight one job's nodes (everything else dimmed) and put the cursor on one of its racks."""
        try:
            self.focus_job = int(job_id)
        except (TypeError, ValueError):  # mirrors SnapshotBuilder._node_job for non-integer ids
            self.focus_job = hash(job_id) & 0x7FFFFFFF
        self.focus_label = str(job_id)
        self.cursor = rack
        if self.zoomed:
            self.action_back()
        self._redraw()

    def action_zoom(self):
        if self._plan is not None and self._plan.rack_aligned and self.snapshot is not None:
            self.zoomed = True
            self.set_class(True, "zoomed")
            self.query_one("#nm-rack").focus()
            self.update_snapshot(self.snapshot)

    def action_back(self):
        if self.zoomed:
            self.zoomed = False
            self.set_class(False, "zoomed")
            self.focus()
            self._redraw()
        elif self.focus_job is not None:
            self.focus_job = None
            self._redraw()

    # -- rendering ---------------------------------------------------------------------------
    def _power_range(self, snap):
        up = (snap.node_state != DOWN) & (snap.node_state != MISSING)
        p = snap.node_power[up] if up.any() else snap.node_power
        return float(p.min()), max(float(p.max()), float(p.min()) + 1.0)

    def _image(self, w, h):
        snap = self.snapshot
        if snap is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        m = snap.meta
        if self._img_size != (w, h) or self._plan is None:
            self._plan = plan_nodemap(m.total_nodes, m.nodes_per_rack, w, h, m.ui_overrides.get("node_map_group"))
            self._pix = pixel_bin_index(self._plan, m.total_nodes, m.nodes_per_rack, w, h)
            self._rack_px = rack_of_pixels(self._plan, w, h, m.total_nodes // m.nodes_per_rack)
            self._img_size = (w, h)
        lo, hi = self._power_range(snap)
        bins = bin_nodes(snap.node_state, snap.node_power, snap.node_job, self._plan.per_bin)
        rgb = bin_colors(bins, MODES[self.mode], lo, hi)
        if self.focus_job is not None:
            frac = self._job_fraction(snap)
            rgb = np.where(frac[:, None] > 0, HIGHLIGHT, (rgb * 0.3).astype(np.uint8))
        img = render_pixels(self._plan, self._pix, rgb)
        if self._plan.rack_aligned:
            self._outline_cursor(img)
        return img

    def _job_fraction(self, snap):
        """Fraction of each bin's nodes that belong to the highlighted job."""
        per_bin = self._plan.per_bin
        mine = (snap.node_job == self.focus_job).astype(np.float32)
        n_bins = -(-len(mine) // per_bin)
        mine = np.concatenate([mine, np.zeros(n_bins * per_bin - len(mine), dtype=np.float32)])
        return mine.reshape(n_bins, per_bin).sum(axis=1)

    def _outline_cursor(self, img):
        """Draw a white frame in the gap around the selected rack tile."""
        p = self._plan
        if p.gap == 0:
            sel = self._rack_px == self.cursor
            ys, xs = np.nonzero(sel)
            if len(ys):
                img[ys, xs] = (img[ys, xs].astype(np.int32) * 0.5 + 127).astype(np.uint8)
            return
        x0 = (self.cursor % p.tiles_x) * (p.tile_w + p.gap)
        y0 = (self.cursor // p.tiles_x) * (p.tile_h + p.gap)
        H, W = img.shape[:2]
        for x in (x0 - 1, x0 + p.tile_w):
            if 0 <= x < W:
                img[max(y0 - 1, 0):min(y0 + p.tile_h + 1, H), x] = 255
        for y in (y0 - 1, y0 + p.tile_h):
            if 0 <= y < H:
                img[y, max(x0 - 1, 0):min(x0 + p.tile_w + 1, W)] = 255

    def _redraw(self):
        self.query_one("#nm-grid", PixelGrid).refresh_image()
        self._info()

    def update_snapshot(self, snap):
        super().update_snapshot(snap)
        self._redraw()
        if self.zoomed:
            self._fill_rack(snap)

    def _info(self):
        snap = self.snapshot
        if snap is None or self._plan is None:
            return
        m, p = snap.meta, self._plan
        if self.focus_job is not None:
            line1 = Text.from_markup(f"[b]job {self.focus_label}[/b]  [#ffaa00]█[/] its nodes, everything else dimmed  "
                                     f"(esc clears, c cycles colors)")
        else:
            line1 = Text.from_markup(f"[b]{MODES[self.mode]}[/b] (c)  " + LEGENDS[MODES[self.mode]])
        n_racks = m.total_nodes // m.nodes_per_rack
        if p.rack_aligned:
            lo, hi = self.cursor * m.nodes_per_rack, (self.cursor + 1) * m.nodes_per_rack
            st = snap.node_state[lo:hi]
            pw = snap.node_power[lo:hi]
            res = (f"{p.per_bin} node{'s' if p.per_bin > 1 else ''}/pixel" if p.per_bin > 1 else "1 node/pixel")
            line2 = (f"{m.total_nodes} nodes, {n_racks} racks, {res} | rack {self.cursor} (nodes {lo}-{hi - 1}): "
                     f"{(st == BUSY).sum()} busy, {(st == DOWN).sum()} down, mean {pw.mean():.0f} W | "
                     f"arrows move, enter zoom")
        else:
            line2 = f"{m.total_nodes} nodes, {p.per_bin} nodes/pixel (terminal too small to tile racks)"
        self.query_one("#nm-info", TextPane).update(Text.assemble(line1, "\n", line2))

    def _fill_rack(self, snap):
        m = snap.meta
        self.query_one("#nm-rack", DataTable).border_title = f"Rack {self.cursor} (left/right: prev/next, esc: back)"
        lo, hi = self.cursor * m.nodes_per_rack, (self.cursor + 1) * m.nodes_per_rack
        jobs = {j[0]: j[1] for j in snap.jobs}
        t = self.query_one("#nm-rack", DataTable)
        colors = {FREE: "dim", BUSY: "green", DOWN: "red", MISSING: "grey30"}
        nodes = range(lo, min(hi, m.total_nodes))
        rows = []
        for n in nodes:
            s, j = int(snap.node_state[n]), int(snap.node_job[n])
            rows.append((str(n), Text(STATE_NAMES[s], style=colors[s]), f"{snap.node_power[n]:.0f}",
                         str(j) if j >= 0 else "", str(jobs.get(j, "")) if j >= 0 else ""))
        sync_table(t, [str(n) for n in nodes], rows)
