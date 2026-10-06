"""Overview: scheduler/power/cooling/network tiles, sparklines, a mini node map and the top jobs."""
import numpy as np
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable
from raps.ui.widgets.textpane import TextPane

from raps.ui.binning import plan_nodemap, pixel_bin_index, bin_nodes, bin_colors, render_pixels
from raps.ui.views.base import View, fmt_dur, fmt_num, kv_lines, sync_table
from raps.ui.widgets.pixelgrid import PixelGrid
from raps.ui.widgets.spark import LabeledSpark

TOP_JOBS = 10


class Overview(View):
    view_key = "1"
    title = "Overview"

    DEFAULT_CSS = """
    Overview #ov-tiles { height: auto; }
    Overview .tile { width: 1fr; height: 7; border: round $primary-darken-2; padding: 0 1; }
    Overview #ov-sparks { height: auto; }
    Overview #ov-bottom { height: 1fr; }
    Overview #ov-map { width: 1fr; border: round $primary-darken-2; }
    Overview #ov-jobs { width: 2fr; border: round $primary-darken-2; }
    Overview.narrow #ov-map { display: none; }
    Overview.narrow #ov-jobs { width: 1fr; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cols = None

    def compose(self) -> ComposeResult:
        with Horizontal(id="ov-tiles"):
            yield TextPane(id="tile-sched", classes="tile")
            yield TextPane(id="tile-power", classes="tile")
            yield TextPane(id="tile-cool", classes="tile")
            yield TextPane(id="tile-net", classes="tile")
        with Vertical(id="ov-sparks"):
            yield LabeledSpark("Power MW", id="sp-power")
            yield LabeledSpark("Util %", id="sp-util")
            yield LabeledSpark("PUE", id="sp-pue")
            yield LabeledSpark("Net util %", id="sp-net")
        with Horizontal(id="ov-bottom"):
            yield PixelGrid(id="ov-map")
            yield DataTable(id="ov-jobs", cursor_type="none", zebra_stripes=True)

    def on_mount(self):
        self.query_one("#ov-map").border_title = "Node map"
        t = self.query_one("#ov-jobs", DataTable)
        t.border_title = "Top jobs"
        self.query_one("#ov-map", PixelGrid).set_source(self._map_image)

    def on_resize(self, event):
        self.set_class(event.size.width < 100, "narrow")

    def _setup(self, snap):
        meta = snap.meta
        self.query_one("#tile-cool").display = meta.has_cooling
        self.query_one("#sp-pue").display = meta.has_cooling
        self.query_one("#tile-net").display = meta.has_network
        self.query_one("#sp-net").display = meta.has_network
        self.query_one("#tile-sched").border_title = "Scheduler"
        self.query_one("#tile-power").border_title = "Power"
        self.query_one("#tile-cool").border_title = "Cooling"
        self.query_one("#tile-net").border_title = "Network"
        t = self.query_one("#ov-jobs", DataTable)
        cols = ["Job", "Name", "Nodes", "Wall"] + (["Slow"] if meta.has_network else [])
        t.add_columns(*cols)
        self._cols = cols

    def _map_image(self, w, h):
        snap = self.snapshot
        if snap is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        m = snap.meta
        plan = plan_nodemap(m.total_nodes, m.nodes_per_rack, w, h)
        bins = bin_nodes(snap.node_state, snap.node_power, snap.node_job, plan.per_bin)
        lo, hi = float(snap.node_power.min()), max(float(snap.node_power.max()), 1.0)
        rgb = bin_colors(bins, "state", lo, hi)
        pix = pixel_bin_index(plan, m.total_nodes, m.nodes_per_rack, w, h)
        return render_pixels(plan, pix, rgb)

    def update_snapshot(self, snap):
        super().update_snapshot(snap)
        if self._cols is None:
            self._setup(snap)
        m = snap.meta
        self.query_one("#tile-sched", TextPane).update(kv_lines([
            ("Time", snap.time_str),
            ("Sim", fmt_dur(snap.sim_elapsed)),
            ("Rate", f"{snap.rate_text}x"),
            ("Jobs", f"{snap.n_running} run / {snap.n_queued} q"),
            ("Nodes", f"{snap.active_nodes} act {snap.free_nodes} free {snap.down_nodes} dn"),
        ]))
        loss_pct = snap.total_loss_mw / snap.total_power_mw * 100 if snap.total_power_mw else 0.0
        self.query_one("#tile-power", TextPane).update(kv_lines([
            ("Util", f"{snap.system_util:.1f}%"),
            ("Power", f"{snap.total_power_mw:.3f} MW"),
            ("Loss", f"{snap.total_loss_mw:.2f} MW {loss_pct:.1f}%"),
            ("PFLOPS", f"{snap.p_flops:.2f}" if snap.p_flops is not None else "n/a"),
            ("GFLOPS/W", f"{snap.g_flops_w:.1f}" if snap.g_flops_w is not None else "n/a"),
        ]))
        if snap.cooling is not None:
            c = snap.cooling.cdu
            t_ret = np.nanmean(c["T_sec_r_C"]) if "T_sec_r_C" in c else float("nan")
            t_sup = np.nanmean(c["T_sec_s_C"]) if "T_sec_s_C" in c else float("nan")
            flow = np.nansum(c["V_flow_sec_GPM"]) if "V_flow_sec_GPM" in c else float("nan")
            self.query_one("#tile-cool", TextPane).update(kv_lines([
                ("PUE", f"{snap.cooling.pue:.3f}" if snap.cooling.pue is not None else "n/a"),
                ("T supply", f"{t_sup:.1f} C"),
                ("T return", f"{t_ret:.1f} C"),
                ("Flow", f"{fmt_num(flow)} gpm"),
                ("Backend", m.cooling_backend + (" (extrapolating)" if snap.cooling.extrapolating else "")),
            ]))
        if snap.network is not None:
            n = snap.network
            self.query_one("#tile-net", TextPane).update(kv_lines([
                ("Topology", m.topology),
                ("Util", f"{n.avg_util * 100:.1f}%"),
                ("Slowdown", f"{n.avg_slowdown:.2f}x"),
                ("Congest", f"{n.congestion_mean:.3f}" if n.congestion_mean is not None else "n/a"),
            ]))
        h = snap.history
        self.query_one("#sp-power", LabeledSpark).set_data(h["power"], f"{snap.total_power_mw:.2f}")
        self.query_one("#sp-util", LabeledSpark).set_data(h["util"], f"{snap.system_util:.1f}")
        if snap.cooling is not None and snap.cooling.pue is not None:
            self.query_one("#sp-pue", LabeledSpark).set_data(h["pue"], f"{snap.cooling.pue:.3f}")
        if snap.network is not None:
            self.query_one("#sp-net", LabeledSpark).set_data(h["net_util"], f"{snap.network.avg_util * 100:.1f}")

        self.query_one("#ov-map", PixelGrid).refresh_image()
        self._fill_jobs(snap)

    def _fill_jobs(self, snap):
        t = self.query_one("#ov-jobs", DataTable)
        # Longest-running first: the most established work is what the operator watches
        top = sorted((j for j in snap.jobs if j[3] == "R"), key=lambda j: -j[6])[:TOP_JOBS]
        rows = []
        for j in top:
            row = [str(j[0]), str(j[1]), str(j[4]), fmt_dur(j[6])]
            if snap.meta.has_network:
                row.append(f"{j[7]:.2f}x")
            rows.append(tuple(Text(c, style="yellow", no_wrap=True) if j[8] else Text(c, no_wrap=True) for c in row))
        sync_table(t, [str(j[0]) for j in top], rows)
