"""Cooling: plant schematic plus a per-CDU heat grid with a selectable metric."""
import numpy as np
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from raps.ui.widgets.textpane import TextPane

from raps.ui.binning import plan_blocks, block_image, heat_colors, group_mean
from raps.ui.views.base import View
from raps.ui.widgets.pixelgrid import PixelGrid
from raps.ui.widgets.schematic import render_schematic, DEFAULT_SPEC

METRIC_ORDER = ["T_sec_r_C", "T_sec_s_C", "T_prim_r_C", "T_prim_s_C", "V_flow_sec_GPM", "V_flow_prim_GPM",
                "p_sec_r_psig", "p_sec_s_psig", "p_prim_r_psig", "p_prim_s_psig", "W_flow_CDUP_kW"]


class Cooling(View):
    view_key = "5"
    title = "Cooling"
    BINDINGS = [
        Binding("m", "metric(1)", "Next metric", show=False),
        Binding("M", "metric(-1)", "Previous metric", show=False),
    ]
    DEFAULT_CSS = """
    Cooling #cl-schem-box { height: auto; max-height: 60%; border: round $primary-darken-2; }
    Cooling #cl-schem { height: auto; padding: 0 1; }
    Cooling #cl-bottom { height: 1fr; min-height: 6; }
    Cooling #cl-grid { border: round $primary-darken-2; width: 3fr; }
    Cooling #cl-side { width: 1fr; min-width: 28; padding: 0 1; border: round $primary-darken-2; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric = 0
        self.metrics = []
        self.spec = DEFAULT_SPEC

    @classmethod
    def applies(cls, meta):
        return meta.has_cooling

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="cl-schem-box"):
            yield TextPane(id="cl-schem")
        with Horizontal(id="cl-bottom"):
            yield PixelGrid(id="cl-grid")
            yield TextPane(id="cl-side")

    def on_mount(self):
        self.query_one("#cl-schem-box").border_title = "Cooling plant"
        self.query_one("#cl-grid", PixelGrid).set_source(self._image)

    def action_metric(self, d: int):
        if self.metrics:
            self.metric = (self.metric + d) % len(self.metrics)
            self._redraw()

    def _current(self):
        snap = self.snapshot
        if snap is None or snap.cooling is None or not self.metrics:
            return None, None
        key = self.metrics[self.metric]
        return key, snap.cooling.cdu[key]

    def _image(self, w, h):
        key, arr = self._current()
        if arr is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        k, group, cols = plan_blocks(len(arr), w, h)
        v = group_mean(arr, group)
        lo, hi = (np.nanmin(arr), np.nanmax(arr)) if np.isfinite(arr).any() else (0.0, 1.0)
        return block_image(heat_colors(v, lo, hi), k, cols, w, h)

    def _redraw(self):
        snap = self.snapshot
        if snap is None or snap.cooling is None:
            return
        self.query_one("#cl-grid", PixelGrid).refresh_image()
        key, arr = self._current()
        label = snap.meta.cdu_labels.get(key, key)
        self.query_one("#cl-grid").border_title = f"CDUs: {label} (m: next metric)"
        t = Text()
        t.append(f"{label}\n", style="bold")
        if np.isfinite(arr).any():
            t.append(f"min  {np.nanmin(arr):.2f}\nmean {np.nanmean(arr):.2f}\nmax  {np.nanmax(arr):.2f}\n")
            t.append(f"CDUs {len(arr)}\n\n")
            t.append("low ")
            for c in ("#440154", "#2a788e", "#22a884", "#7ad151", "#fde725"):
                t.append("█", style=c)
            t.append(" high\n")
        c = snap.cooling
        t.append("\n")
        t.append(f"Backend: {snap.meta.cooling_backend}\n", style="dim")
        if c.extrapolating:
            t.append("Inputs outside the surrogate's training range (clipped)\n", style="bold yellow")
        self.query_one("#cl-side", TextPane).update(t)

    def update_snapshot(self, snap):
        super().update_snapshot(snap)
        if snap.cooling is None:
            return
        if not self.metrics:
            self.metrics = [k for k in METRIC_ORDER if k in snap.cooling.cdu] + \
                           [k for k in snap.cooling.cdu if k not in METRIC_ORDER]
            override = snap.meta.ui_overrides.get("cooling_schematic")
            if override:
                self.spec = override
        width = self.query_one("#cl-schem-box").size.width - 4
        self.query_one("#cl-schem", TextPane).update(render_schematic(snap, width, self.spec))
        self._redraw()
