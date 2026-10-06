"""Network: topology summary, utilization/congestion/slowdown trends, hottest links, per-job slowdown."""
import numpy as np
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable
from raps.ui.widgets.textpane import TextPane

from raps.ui.binning import block_image, heat_colors, plan_blocks
from raps.ui.views.base import View, kv_lines
from raps.ui.widgets.pixelgrid import PixelGrid
from raps.ui.widgets.spark import LabeledSpark

BARS = " ▁▂▃▄▅▆▇█"


class Network(View):
    view_key = "6"
    title = "Network"
    DEFAULT_CSS = """
    Network #nw-top { height: 7; }
    Network .tile { width: 1fr; height: 7; border: round $primary-darken-2; padding: 0 1; }
    Network #nw-sparks { height: 3; }
    Network #nw-main { height: 1fr; }
    Network #nw-left { width: 1fr; }
    Network #nw-right { width: 1fr; }
    Network DataTable { height: 1fr; border: round $primary-darken-2; }
    Network #nw-hist { height: 5; border: round $primary-darken-2; padding: 0 1; }
    Network #nw-df { border: round $primary-darken-2; height: 1fr; display: none; }
    Network.dragonfly #nw-df { display: block; }
    Network.narrow #nw-right { display: none; }
    """

    @classmethod
    def applies(cls, meta):
        return meta.has_network

    def compose(self) -> ComposeResult:
        with Horizontal(id="nw-top"):
            yield TextPane(id="nw-topo", classes="tile")
            yield TextPane(id="nw-load", classes="tile")
        with Vertical(id="nw-sparks"):
            yield LabeledSpark("Util %", id="nw-sp-util")
            yield LabeledSpark("Congestion", id="nw-sp-cong")
            yield LabeledSpark("Slowdown x", id="nw-sp-slow")
        with Horizontal(id="nw-main"):
            with Vertical(id="nw-left"):
                yield TextPane(id="nw-hist")
                yield DataTable(id="nw-links", cursor_type="none", zebra_stripes=True)
            with Vertical(id="nw-right"):
                yield DataTable(id="nw-jobs", cursor_type="none", zebra_stripes=True)
                yield PixelGrid(id="nw-df")

    def on_mount(self):
        self.query_one("#nw-topo").border_title = "Topology"
        self.query_one("#nw-load").border_title = "Load"
        self.query_one("#nw-hist").border_title = "Link utilization histogram"
        links = self.query_one("#nw-links", DataTable)
        links.border_title = "Most congested links"
        links.add_columns("Link", "Util")
        jobs = self.query_one("#nw-jobs", DataTable)
        jobs.border_title = "Most slowed-down jobs"
        jobs.add_columns("Job", "Nodes", "Slowdown", "Dilated")
        self.query_one("#nw-df", PixelGrid).set_source(self._df_image)
        self.query_one("#nw-df").border_title = "Dragonfly groups: peak util among top links"

    def on_resize(self, event):
        self.set_class(event.size.width < 90, "narrow")

    def _df_image(self, w, h):
        net = self.snapshot.network if self.snapshot else None
        if net is None or net.group_matrix is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        m = net.group_matrix
        n = m.shape[0]
        k, group, cols = plan_blocks(n * n, w, h)
        flat = m.ravel()
        hi = max(float(flat.max()), 1e-9)
        rgb = heat_colors(np.where(flat > 0, flat, np.nan), 0.0, hi)
        rgb[flat <= 0] = (38, 44, 52)
        # keep rows of the matrix aligned: blocks per row must be n
        if group == 1 and cols >= n:
            grid = rgb.reshape(n, n, 3)
            padded = np.zeros((n, cols, 3), dtype=np.uint8)
            padded[:, :n] = grid
            return block_image(padded.reshape(-1, 3), k, cols, w, h)
        return block_image(rgb, k, cols, w, h)

    def update_snapshot(self, snap):
        super().update_snapshot(snap)
        net = snap.network
        if net is None:
            return
        m = snap.meta
        self.set_class(m.topology == "dragonfly", "dragonfly")
        stats = net.link_stats
        self.query_one("#nw-topo", TextPane).update(kv_lines([
            ("Topology", m.topology),
            ("Links", stats["num_links"] if stats and "num_links" in stats else "n/a"),
            ("Link max", f"{stats['max']:.3f}" if stats else "n/a"),
            ("Link mean", f"{stats['mean']:.3f}" if stats else "n/a"),
            ("Link std", f"{stats['std_dev']:.3f}" if stats else "n/a"),
        ]))
        self.query_one("#nw-load", TextPane).update(kv_lines([
            ("Avg util", f"{net.avg_util * 100:.1f}%"),
            ("Avg tx", f"{net.avg_tx:.3g} B"),
            ("Avg rx", f"{net.avg_rx:.3g} B"),
            ("Slowdown", f"{net.avg_slowdown:.2f}x"),
            ("Congest", f"{net.congestion_mean:.3f}" if net.congestion_mean is not None else "n/a"),
        ]))
        h = snap.history
        self.query_one("#nw-sp-util", LabeledSpark).set_data(h["net_util"], f"{net.avg_util * 100:.1f}")
        self.query_one("#nw-sp-cong", LabeledSpark).set_data(h["congestion"], f"{net.congestion_mean or 0:.3f}")
        self.query_one("#nw-sp-slow", LabeledSpark).set_data(h["slowdown"], f"{net.avg_slowdown:.2f}")

        hist = self.query_one("#nw-hist", TextPane)
        if stats and stats.get("histogram"):
            counts = np.array(stats["histogram"], dtype=float)
            top = max(counts.max(), 1.0)
            bars = "".join(BARS[int(round(c / top * (len(BARS) - 1)))] * 3 for c in counts)
            edge = max(1.0, float(stats["max"]))
            hist.update(Text.assemble(bars, "\n", f"0{' ' * (len(bars) - 8)}{edge:.2f}\n",
                                      ("links per utilization bin (tallest = ", "dim"),
                                      (f"{int(counts.max())}", "dim"), (")", "dim")))
        else:
            hist.update(Text("no link data yet", style="dim"))

        links = self.query_one("#nw-links", DataTable)
        links.clear()
        if stats:
            for (u, v), util in stats["top_links"][:10]:
                links.add_row(f"{u} - {v}", f"{util:.3f}")
        jobs = self.query_one("#nw-jobs", DataTable)
        jobs.clear()
        for jid, nodes, slow, dil in net.jobs[:30]:
            style = "yellow" if dil else None
            jobs.add_row(Text(str(jid), style=style), Text(str(nodes), style=style),
                         Text(f"{slow:.2f}x", style=style), Text("yes" if dil else "", style=style))
        self.query_one("#nw-df", PixelGrid).refresh_image()
