"""Modal showing one job in detail: summary, placement and power history."""
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen

from raps.ui.views.base import fmt_dur
from raps.ui.widgets.spark import LabeledSpark
from raps.ui.widgets.textpane import TextPane

STATE_NAMES = {"R": "running", "PD": "queued", "C": "completed", "Cing": "completing",
               "CA": "cancelled", "F": "failed", "TO": "timed out"}


def fmt_ranges(ranges, limit=6):
    """[(0, 15), (20, 20)] -> '0-15, 20', eliding past `limit` ranges."""
    parts = [str(a) if a == b else f"{a}-{b}" for a, b in ranges[:limit]]
    if len(ranges) > limit:
        parts.append(f"... (+{len(ranges) - limit} more ranges)")
    return ", ".join(parts)


def fmt_rel(t):
    """Sim time relative to the start of the run, e.g. 'T+01:02:03' or 'T-00:10:00' for earlier."""
    if t is None:
        return "n/a"
    return ("T+" if t >= 0 else "T-") + fmt_dur(abs(t))


def fmt_power(w):
    if w is None:
        return "n/a"
    return f"{w / 1000:.1f} kW" if w < 1e6 else f"{w / 1e6:.2f} MW"


def summary_text(d, ended=False):
    state = STATE_NAMES.get(d.state, d.state)
    t = Text()
    if ended:
        t.append("Job is no longer running or queued; showing the last known state\n", style="bold yellow")
    rows = [("State", state), ("Account", d.account), ("Nodes", f"{d.nodes_required:,}")]
    for k, v in rows:
        t.append(f"{k + ':':<11}", style="dim")
        t.append(f"{v}\n", style="bold")
    t.append(f"{'Submitted:':<11}", style="dim")
    t.append(f"{fmt_rel(d.submit_s)}\n")
    if d.state == "R":
        t.append(f"{'Started:':<11}", style="dim")
        t.append(f"{fmt_rel(d.start_s)}\n")
    if d.power_now_w is not None:
        t.append(f"{'Power:':<11}", style="dim")
        t.append(f"now {fmt_power(d.power_now_w)}, avg {fmt_power(d.power_avg_w)}, "
                 f"peak {fmt_power(d.power_peak_w)}\n")
    if d.dilated:
        t.append(f"{'Network:':<11}", style="dim")
        t.append(f"dilated by congestion, slowdown {d.slowdown:.2f}x\n", style="yellow")
    return t


def progress_text(d, width=30):
    if d.state != "R":
        return Text("Waiting in the queue" if d.state == "PD" else "")
    t = Text()
    t.append(f"{'Elapsed:':<11}", style="dim")
    if d.limit_s > 0:
        frac = min(d.run_s / d.limit_s, 1.0)
        filled = int(width * frac)
        t.append("█" * filled, style="green")
        t.append("░" * (width - filled), style="dim")
        t.append(f" {fmt_dur(d.run_s)} of {fmt_dur(d.limit_s)} ({frac * 100:.0f}%)")
    else:
        t.append(f"{fmt_dur(d.run_s)} (no time limit)")
    return t


def placement_text(d):
    t = Text()
    if not d.node_ranges:
        return t
    t.append(f"{'Racks:':<11}", style="dim")
    t.append(f"{fmt_ranges([(r, r) for r in d.racks])}  ({len(d.racks)} rack{'s' if len(d.racks) != 1 else ''})\n")
    t.append(f"{'Node ids:':<11}", style="dim")
    t.append(fmt_ranges(d.node_ranges))
    return t


class JobDetailScreen(ModalScreen):
    BINDINGS = [
        Binding("escape,q,enter", "app.pop_screen", "Close"),
        Binding("m", "show_in_map", "Highlight in node map"),
    ]
    DEFAULT_CSS = """
    JobDetailScreen { align: center middle; }
    JobDetailScreen > Vertical { width: 90; max-width: 100%; height: auto; max-height: 100%;
                                 border: round $primary; background: $surface; padding: 1 2; }
    JobDetailScreen #jd-hint { color: $text-muted; margin-top: 1; }
    JobDetailScreen #jd-power { margin-top: 1; }
    JobDetailScreen #jd-place { margin-top: 1; }
    """

    def __init__(self, job_id):
        super().__init__()
        self.job_id = job_id
        self.detail = None       # last known JobDetail
        self.ended = False

    def compose(self) -> ComposeResult:
        with Vertical(id="jd-box"):
            yield TextPane("loading...", id="jd-summary")
            yield TextPane(id="jd-progress")
            yield LabeledSpark("Power", id="jd-power")
            yield TextPane(id="jd-place")
            yield TextPane("m highlight in node map   esc close", id="jd-hint")

    def on_mount(self):
        self.query_one("#jd-box").border_title = f"Job {self.job_id}"
        snap = getattr(self.app, "_applied", None)
        if snap is not None:
            self.update_snapshot(snap)

    def update_snapshot(self, snap):
        d = snap.job_detail
        if d is not None and str(d.id) != str(self.job_id):
            d = None
        if d is None:
            if self.detail is None:
                self.query_one("#jd-summary", TextPane).update("Waiting for job data...")
                return
            self.ended = True
            d = self.detail
        else:
            self.detail, self.ended = d, False
        self.query_one("#jd-box").border_title = f"Job {d.id}: {d.name}"
        self.query_one("#jd-summary", TextPane).update(summary_text(d, self.ended))
        self.query_one("#jd-progress", TextPane).update(progress_text(d))
        spark = self.query_one("#jd-power", LabeledSpark)
        spark.display = bool(d.power_w)
        if d.power_w:
            spark.set_data(d.power_w, fmt_power(d.power_now_w))
        self.query_one("#jd-place", TextPane).update(placement_text(d))

    def on_unmount(self):
        builder = getattr(self.app, "builder", None)
        if builder is not None:
            builder.selected_job = None

    def action_show_in_map(self):
        d = self.detail
        self.app.pop_screen()
        if d is not None and d.racks:
            self.app.show_job_in_map(d.id, d.racks[0])
