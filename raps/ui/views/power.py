"""Power: rack power heatmap, totals, and the hottest racks and CDUs."""
import numpy as np
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable
from raps.ui.widgets.textpane import TextPane

from raps.ui.binning import plan_blocks, block_image, heat_colors, group_mean
from raps.ui.views.base import View
from raps.ui.widgets.pixelgrid import PixelGrid

TOP_N = 12


class Power(View):
    view_key = "4"
    title = "Power"
    DEFAULT_CSS = """
    Power #pw-tiles { height: 3; padding: 0 1; }
    Power #pw-main { height: 1fr; }
    Power #pw-left { width: 2fr; }
    Power #pw-grid { border: round $primary-darken-2; }
    Power #pw-legend { height: 1; padding: 0 1; }
    Power #pw-right { width: 1fr; min-width: 30; }
    Power DataTable { height: 1fr; border: round $primary-darken-2; }
    Power.narrow #pw-right { display: none; }
    """

    def compose(self) -> ComposeResult:
        yield TextPane(id="pw-tiles")
        with Horizontal(id="pw-main"):
            with Vertical(id="pw-left"):
                yield PixelGrid(id="pw-grid")
                yield TextPane(id="pw-legend")
            with Vertical(id="pw-right"):
                yield DataTable(id="pw-racks", cursor_type="none", zebra_stripes=True)
                yield DataTable(id="pw-cdus", cursor_type="none", zebra_stripes=True)

    def on_mount(self):
        self.query_one("#pw-grid", PixelGrid).set_source(self._image)
        self.query_one("#pw-grid").border_title = "Rack power (kW)"
        r = self.query_one("#pw-racks", DataTable)
        r.border_title = "Hottest racks"
        r.add_columns("Rack", "CDU", "kW", "Loss kW")
        c = self.query_one("#pw-cdus", DataTable)
        c.border_title = "Hottest CDUs"
        c.add_columns("CDU", "kW", "Loss kW")

    def on_resize(self, event):
        self.set_class(event.size.width < 90, "narrow")

    def _image(self, w, h):
        snap = self.snapshot
        if snap is None:
            return np.zeros((h, w, 3), dtype=np.uint8)
        p = snap.rack_power.ravel()
        k, group, cols = plan_blocks(len(p), w, h)
        v = group_mean(p, group)
        lo, hi = float(np.min(p)), float(np.max(p))
        return block_image(heat_colors(v, lo, hi), k, cols, w, h)

    def update_snapshot(self, snap):
        super().update_snapshot(snap)
        p = snap.rack_power
        loss_pct = snap.total_loss_mw / snap.total_power_mw * 100 if snap.total_power_mw else 0.0
        t = Text()
        t.append("Total ", style="dim")
        t.append(f"{snap.total_power_mw:.3f} MW", style="bold")
        t.append("  Loss ", style="dim")
        t.append(f"{snap.total_loss_mw:.3f} MW ({loss_pct:.1f}%)", style="bold")
        t.append("  PFLOPS ", style="dim")
        t.append(f"{snap.p_flops:.2f}" if snap.p_flops is not None else "n/a", style="bold")
        t.append("  GFLOPS/W ", style="dim")
        t.append(f"{snap.g_flops_w:.1f}" if snap.g_flops_w is not None else "n/a", style="bold")
        t.append("\n")
        t.append("Racks ", style="dim")
        t.append(f"{p.size}", style="bold")
        t.append("  kW min/mean/max ", style="dim")
        t.append(f"{p.min():.1f} / {p.mean():.1f} / {p.max():.1f}", style="bold")
        t.append("  Util ", style="dim")
        t.append(f"{snap.system_util:.1f}%", style="bold")
        self.query_one("#pw-tiles", TextPane).update(t)
        self.query_one("#pw-legend", TextPane).update(
            f"low [#440154]█[/][#2a788e]█[/][#22a884]█[/][#7ad151]█[/][#fde725]█[/] high "
            f"({p.min():.0f} to {p.max():.0f} kW); racks in CDU order, wrapped to the window")
        self.query_one("#pw-grid", PixelGrid).refresh_image()

        racks = self.query_one("#pw-racks", DataTable)
        racks.clear()
        flat = p.ravel()
        r = snap.meta.racks_per_cdu
        for i in np.argsort(flat)[::-1][:TOP_N]:
            racks.add_row(str(i + 1), str(i // r + 1), f"{flat[i]:.1f}", f"{snap.rack_loss.ravel()[i]:.1f}")
        cdus = self.query_one("#pw-cdus", DataTable)
        cdus.clear()
        for i in np.argsort(snap.cdu_power)[::-1][:TOP_N]:
            cdus.add_row(str(i + 1), f"{snap.cdu_power[i]:.1f}", f"{snap.cdu_loss[i]:.1f}")
