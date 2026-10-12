"""The console's tabs. Each view renders from the latest ``ClusterState`` when it is visible."""
import codecs
import fcntl
import os
import pty
import re
import shlex
import signal
import struct
import subprocess
import tempfile
import termios
import threading
from datetime import datetime, timedelta
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.suggester import SuggestFromList
from textual.widgets import DataTable, Input, RichLog

from raps.live.analysis import (
    eligible, forecast_summary, job_power, node_job_map, queue_position, reason_counts, size_histogram, summarize,
    top_accounts,
)
from raps.live.forecast import remaining
from raps.live.poller import cache_dir
from raps.live.widgets import (
    AMBER, CYAN, GREEN, PINK, RED, SLATE, VIOLET, NodeMap, Tile, bar, gradient, heat, human_dur,
    human_num, lerp_hex, rel_time, spark, state_text,
)
from raps.ui.views.base import sync_table
from raps.ui.widgets.textpane import TextPane

MUTED = "#94a3b8"


class LiveView(Vertical):
    """One tab. ``render_state`` runs on every new state while visible, and on becoming visible."""

    DEFAULT_CSS = """
    LiveView { width: 1fr; height: 1fr; }
    LiveView .panel { border: round $primary-darken-2; border-title-color: $secondary;
                      border-title-style: bold; padding: 0 1; background: $panel; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = None
        self.trend = None

    def apply(self, state, trend):
        self.state, self.trend = state, trend
        if self.is_on_screen and self.state is not None:
            self.render_state(state)

    def render_state(self, state):
        pass

    def on_show(self):
        if self.state is not None:
            self.render_state(self.state)

    @property
    def is_on_screen(self):
        return self.display and self.parent is not None and self.parent.display


def _panel(widget, title):
    widget.add_class("panel")
    widget.border_title = title
    return widget


def _waiting(what):
    return Text(f"waiting for {what} …", style="italic #64748b")


# ---------------------------------------------------------------------------------------------
# 1 · Overview
# ---------------------------------------------------------------------------------------------


BASIS_STYLE = {"job": (f"bold {GREEN}", "●"), "user": (CYAN, "◐"), "project": (VIOLET, "○")}


def forecast_text(state, job, long=False):
    """Forecast runtime (pending) or time left (running) from the history rule, marked by its basis."""
    if state.forecast is None:
        return Text("…", style=SLATE)
    p = state.predict(job)
    if p is None:
        return Text("no history", style=SLATE) if long else Text("—", style=SLATE)
    style, mark = BASIS_STYLE[p.basis]
    if job.running:
        left = remaining(job, p)
        if left is None:
            t = Text(f"{mark} past forecast", style=AMBER)
        else:
            t = Text(f"{mark} {human_dur(left)} left", style=style)
            if long:
                t.append(f" · ends ~{(datetime.now() + timedelta(seconds=left)):%H:%M}",
                         style=MUTED)
    else:
        t = Text(f"{mark} {human_dur(p.runtime)}", style=style)
        if long:
            t.append(f" · {p.runtime / job.limit * 100:.0f}% of limit", style=MUTED)
    if long:
        t.append(f" · {p.basis} ×{p.n}", style=MUTED)
    return t


def twin_start(state, job):
    """(text, datetime or None) for the RAPS twin's start forecast of a pending job."""
    tw = state.twin
    if tw is None or not job.pending:
        return None
    if job.id not in tw.starts:
        return Text("not simulated", style=SLATE)  # blocked: dependency, hold, limits
    start = tw.starts[job.id]
    if start is None:
        return Text("> 14 days", style=SLATE)
    now = datetime.now()
    return Text("now" if start <= now + timedelta(minutes=1) else rel_time(start, now), style=f"bold {CYAN}")


def watts_style(w, max_w):
    return heat(min(w / max_w, 1))


def power_text(state, job):
    """History-rule forecast of mean watts per node."""
    p = state.predict(job)
    if p is None or p.power is None:
        return Text("—", justify="right", style=SLATE)
    return Text(f"{p.power:,.0f}", justify="right", style=watts_style(p.power, state.node_max_w))


def power_now_text(state, job):
    """Measured watts per node right now (running jobs)."""
    w = job_power(job, state.node_power)
    if w is None:
        return Text("—", justify="right", style=SLATE)
    return Text(f"{w:,.0f}", justify="right", style=f"bold {watts_style(w, state.node_max_w)}")


def job_energy(state, job):
    """
    (kWh, basis) the job uses over its whole run: nodes x watts per node x runtime, or None without a power figure.
    Runtime is the history rule's forecast (at least the time already run), else the time limit (basis "limit", an
    upper bound). Watts per node is what the job draws now if it runs, else the history rule's forecast.
    """
    p = state.predict(job)
    watts = job_power(job, state.node_power)
    if watts is None and p is not None:
        watts = p.power
    if watts is None:
        return None
    if p is not None:
        runtime, basis = (max(p.runtime, job.elapsed) if job.running else p.runtime), p.basis
    elif job.limit:
        runtime, basis = job.limit, "limit"
    else:
        return None
    return job.nodes * watts * runtime / 3.6e6, basis


def energy_text(est):
    """'1.2 MWh' / '340 kWh', colored by what the runtime forecast rests on (muted: only the time limit)."""
    if est is None:
        return Text("—", justify="right", style=SLATE)
    kwh, basis = est
    style = BASIS_STYLE[basis][0] if basis in BASIS_STYLE else MUTED
    num = f"{kwh / 1000:,.2f} MWh" if kwh >= 1000 else f"{kwh:,.0f} kWh"
    return Text(("≤" if basis == "limit" else "") + num, justify="right", style=style)


def peak_low(tw, hours=24, step=300):
    """((watts, seconds) at the forecast peak, at the low) over the next `hours`."""
    pts = [(tw.power_at(t), t) for t in range(0, hours * 3600 + 1, step)]
    pts = [p for p in pts if p[0] is not None]
    return (max(pts), min(pts)) if pts else (None, None)


class Overview(LiveView):
    DEFAULT_CSS = """
    Overview #ov-tiles { height: 7; }
    Overview #ov-body { height: 1fr; }
    Overview #ov-mapbox { width: 3fr; }
    Overview #ov-map { height: 1fr; }
    Overview #ov-legend { height: 2; }
    Overview #ov-right { width: 2fr; min-width: 44; }
    Overview #ov-right > TextPane { height: auto; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="ov-tiles"):
            yield Tile("utilization", id="t-util")
            yield Tile("running jobs", id="t-run", classes="good")
            yield Tile("pending jobs", id="t-pend", classes="warn")
            yield Tile("backlog node-h", id="t-backlog")
            yield Tile("system power MW", id="t-power", classes="accent")
            yield Tile("my jobs", id="t-mine", classes="accent")
            yield Tile("nodes out", id="t-out", classes="bad")
        with Horizontal(id="ov-body"):
            with Vertical(id="ov-mapbox") as box:
                _panel(box, "machine")
                yield NodeMap(id="ov-map")
                yield TextPane("", id="ov-legend")
            with VerticalScroll(id="ov-right"):
                yield _panel(TextPane(_waiting("sinfo"), id="ov-parts"), "partitions")
                yield _panel(TextPane(_waiting("squeue"), id="ov-sizes"), "queue by job size")
                yield _panel(TextPane(_waiting("RAPS twin"), id="ov-twin"), "RAPS twin · next 24 h")
                yield _panel(TextPane(_waiting("squeue"), id="ov-next"), "up next · large jobs · twin vs slurm")
                yield _panel(TextPane(_waiting("squeue"), id="ov-accts"), "top projects · running nodes · pending node-h")
                yield _panel(TextPane(_waiting("samples"), id="ov-trend"), "trends")

    def render_state(self, state):
        s = summarize(state)
        jobs = state.jobs or []
        w = max(self.query_one("#ov-right").size.width - 4, 30)

        fs = forecast_summary(state) if state.forecast is not None else None
        util = s.utilization
        # each caption comes in several lengths; a tile shows the longest that fits its width
        nodes = Text(f" {s.busy_nodes:,}/{s.usable_nodes:,}", style=MUTED)
        self.query_one("#t-util", Tile).set(f"{util * 100:.1f}%", [bar(util, 20, GREEN, CYAN) + nodes,
                                                                    bar(util, 10, GREEN, CYAN) + nodes, nodes])
        run = [Text(f"{s.run_nodes:,} nodes", style=MUTED)]
        if fs is not None:
            free = (f"{human_num(fs.free_1h_fcst)} free <1h", f"bold {GREEN}")
            run = [Text.assemble((f"{s.run_nodes:,} nodes · ", MUTED), free), Text.assemble(free)] + run
        self.query_one("#t-run", Tile).set(f"{s.run_jobs:,}", run)
        self.query_one("#t-pend", Tile).set(f"{s.pend_jobs:,}", [
            Text.assemble((f"{s.eligible_jobs:,} eligible", AMBER), (f" · {s.blocked_jobs:,} blocked", MUTED)),
            Text.assemble((f"{s.eligible_jobs:,} elig", AMBER), (f" · {s.blocked_jobs:,} blkd", MUTED))])
        backlog = [Text(f"≈ {s.drain_hours:.1f} h of full machine", style=MUTED),
                   Text(f"≈ {s.drain_hours:.1f} h of machine", style=MUTED)]
        if fs is not None and s.usable_nodes:
            # hours of the full machine: at the requested limits, and at the forecast runtimes
            fh = fs.backlog_nh / s.usable_nodes
            fc = (f" · {fh:.1f} h fcst", f"bold {GREEN}")
            backlog = [Text.assemble((f"{s.drain_hours:.1f} h at limits", MUTED), fc),
                       Text.assemble((f"{s.drain_hours:.1f} h limit", MUTED), fc),
                       Text(f"{fh:.1f} h machine (fcst)", style=f"bold {GREEN}"),
                       Text(f"{fh:.1f} h fcst", style=f"bold {GREEN}")]
        self.query_one("#t-backlog", Tile).set(human_num(s.backlog_nh), backlog)
        self.query_one("#t-mine", Tile).set(f"{s.my_run}", [
            Text.assemble(("running", PINK), (f" · {s.my_pend} pending · {s.my_nodes:,} nodes", MUTED)),
            Text.assemble(("run", PINK), (f" · {s.my_pend} pend · {s.my_nodes:,} n", MUTED))])
        if state.power is not None:
            share = state.power / max(state.machine_max_w, 1) * 100
            mx = state.machine_max_w / 1e6
            power = [Text(f"{share:.0f}% of max {mx:.1f} MW", style=MUTED), Text(f"{share:.0f}% of max", style=MUTED)]
            if state.twin is not None and state.twin.power_now is not None:
                (pk, tpk), _ = peak_low(state.twin)
                when, short = ("now", "now") if tpk == 0 else (f"in {human_dur(tpk)}", f"+{human_dur(tpk)}")
                power = [Text.assemble((f"{share:.0f}% of max · ", MUTED), (f"peak {pk / 1e6:.1f} MW {when}", PINK)),
                         Text.assemble((f"{share:.0f}% max · ", MUTED), (f"peak {pk / 1e6:.1f} {when}", PINK)),
                         Text(f"peak {pk / 1e6:.1f} MW {when}", style=PINK),
                         Text(f"pk {pk / 1e6:.1f} {short}", style=PINK)]
            self.query_one("#t-power", Tile).set(f"{state.power / 1e6:.2f}", power)
        out_frac = s.out_nodes / s.total_nodes if s.total_nodes else 0
        self.query_one("#t-out", Tile).set(f"{s.out_nodes}", [
            Text(f"{out_frac * 100:.1f}% · {s.planned_nodes} planned", style=MUTED),
            Text(f"{s.planned_nodes} planned", style=MUTED)])

        nm = self.query_one("#ov-map", NodeMap)
        if state.node_states:
            nm.set_data(state.node_states, node_job_map(jobs), state.user, node_power=state.node_power,
                        max_w=state.node_max_w, **node_access(state))
            hovered = getattr(self, "hovered", None)
            self.query_one("#ov-legend", TextPane).update(map_footer(nm, hovered, "hover a node · click for its job"))

        if state.partitions:
            t = Text()
            for i, p in enumerate(state.partitions):
                frac = p.alloc / p.total if p.total else 0
                if i:
                    t.append("\n")
                t.append(f"{p.name + ('*' if p.default else ''):<13}", style="bold")
                t.append_text(bar(frac, max(w - 34, 6), lerp_hex(GREEN, CYAN, 0.2), CYAN))
                t.append(f" {frac * 100:5.1f}%", style=heat(frac))
                t.append(f" {p.idle:>5,} idle", style=MUTED)
            self.query_one("#ov-parts", TextPane).update(t)

        if state.jobs is not None:
            rows = size_histogram(jobs)
            rpeak = max(max(r[2] for r in rows), 1)
            ppeak = max(max(r[4] for r in rows), 1)
            bw = max((w - 26) // 2, 4)
            t = Text.assemble(("nodes      ", MUTED), (f"{'running (nodes)':<{bw + 6}}", GREEN), ("waiting (nodes)", AMBER))
            for label, rj, rn, pj, pn in rows:
                t.append(f"\n{label:>9} ", style="bold")
                t.append_text(bar(rn / rpeak, bw, GREEN, CYAN))
                t.append(f" {rj:>4} ", style=GREEN)
                t.append_text(bar(pn / ppeak, bw, AMBER, RED))
                t.append(f" {pj:>4}", style=AMBER)
            self.query_one("#ov-sizes", TextPane).update(t)

            self.query_one("#ov-next", TextPane).update(up_next(state, w))
            if state.twin is not None:
                self.query_one("#ov-twin", TextPane).update(twin_panel(state, w))

            accts = top_accounts(jobs, n=7)
            peak = max((a[1] for a in accts), default=1) or 1
            t = Text()
            for i, (name, rn, rj, pnh) in enumerate(accts):
                if i:
                    t.append("\n")
                mine = name in {j.account for j in state.my_jobs}
                t.append(f"{name:<10}", style=f"bold {PINK}" if mine else "bold")
                t.append_text(bar(rn / peak, max(w - 32, 6), VIOLET, CYAN))
                t.append(f" {rn:>5,}n {rj:>3}j", style=MUTED)
                t.append(f" {human_num(pnh):>6}", style=AMBER)
            self.query_one("#ov-accts", TextPane).update(t)

        tr = self.trend
        if tr is not None and tr.samples:
            sw = max(w - 26, 10)
            span = human_dur(tr.span_seconds()) if tr.span_seconds() else "now"
            t = Text(f"last {span} · {len(tr.samples)} samples\n", style=MUTED)
            for label, key, color in (("running nodes", "run_nodes", GREEN), ("pending jobs", "pend_jobs", AMBER),
                                      ("backlog node-h", "backlog_nh", VIOLET), ("system MW", "power_mw", PINK),
                                      ("my jobs (run)", "my_run", PINK)):
                vals = tr.series(key)
                if not vals:
                    continue
                t.append(f"{label:<15}", style=MUTED)
                t.append_text(spark(vals, sw, color))
                last = f"{vals[-1]:.2f}" if key == "power_mw" else human_num(vals[-1])
                t.append(f" {last:>7}\n", style=f"bold {color}")
            t.rstrip()
            self.query_one("#ov-trend", TextPane).update(t)

    def action_color_mode(self):
        nm = self.query_one("#ov-map", NodeMap)
        nm.mode = NodeMap.MODES[(NodeMap.MODES.index(nm.mode) + 1) % len(NodeMap.MODES)]
        if self.state is not None:
            self.render_state(self.state)

    @on(NodeMap.Hover)
    def _hover(self, event):
        self.hovered = event.nodes  # kept so refreshes show the node instead of the legend
        legend = self.query_one("#ov-legend", TextPane)
        nm = self.query_one("#ov-map", NodeMap)
        legend.update(map_footer(nm, event.nodes, "hover a node · click for its job"))

    @on(NodeMap.Clicked)
    def _click(self, event):
        self.app.open_node_job(event.nodes)


def up_next(state, width, n=6):
    """The next big jobs to start (RAPS twin, else Slurm's estimate), and reservations about to begin."""
    now = datetime.now()
    starts = state.twin.starts if state.twin is not None else {}

    def when(j):
        return starts.get(j.id) or j.start

    jobs = sorted((j for j in state.jobs or [] if j.pending and j.nodes >= 64 and when(j) and when(j) > now),
                  key=when)[:n]
    t = Text()
    for i, j in enumerate(jobs):
        if i:
            t.append("\n")
        mine = j.user == state.user
        t.append(f"{rel_time(when(j), now):>10} ", style=f"bold {CYAN}")
        t.append(f"{j.nodes:>5,}n ", style=f"bold {heat(min(j.nodes / 9408, 1) ** 0.4)}")
        t.append(f"{j.user[:10]:<11}", style=f"bold {PINK}" if mine else "")
        if starts:
            t.append(f"slurm {rel_time(j.start, now) if j.start else '—':<10} ", style=MUTED)
        t.append(f"{j.name[:max(width - 56, 6)]}", style=MUTED)
    if not jobs:
        t.append("no large jobs with a start estimate", style=MUTED)
    soon = sorted((r for r in state.reservations or [] if r.start and r.start > now
                   and (r.start - now).days < 14 and r.nodes >= 8), key=lambda r: r.start)[:3]
    for r in soon:
        t.append("\n⚠ ", style=f"bold {AMBER}")
        t.append(f"reservation {rel_time(r.start, now)}: ", style=AMBER)
        t.append(f"{r.nodes:,} nodes ", style="bold")
        t.append(f"{r.name[:max(width - 40, 8)]}", style=MUTED)
    return t


NODES_HINT = "hover a node, or arrows · [ ] cabinet · shift+↑↓ cabinet row · home/end · enter: its job · esc clears"


def map_footer(nm, nodes, hint):
    """Two lines under a node map: the color legend, then the hovered/highlighted node (or a hint)."""
    t = nm.legend()
    t.append("\n")
    t.append_text(node_hover_text(nodes, nm) if nodes else Text(hint, style="#64748b"))
    return t


_NODE_PARTS = {}


def node_access(state):
    """NodeMap kwargs: each node's partitions (cached per layout) and which I may submit to."""
    out = {}
    if state.layout:
        key = id(state.layout)
        if key not in _NODE_PARTS:
            _NODE_PARTS.clear()
            parts = {}
            for name, part in state.layout.items():
                for n in part.nodes:
                    parts.setdefault(n, []).append(name)
            _NODE_PARTS[key] = parts
        out["node_parts"] = _NODE_PARTS[key]
    if state.access is not None:
        out["part_ok"] = {p: state.access.ok(p) for _, p in state.access.results}
    return out


def node_hover_text(nodes, nm):
    name = nodes[0]
    t = Text()
    t.append(f"{name}" + (f" (+{len(nodes) - 1})" if len(nodes) > 1 else ""), style=f"bold {CYAN}")
    t.append(f"  {nm.node_states.get(name, '?')}", style=MUTED)
    w = nm.node_power.get(name)
    if w is not None:
        t.append(f"  {w:,} W ({w / nm.max_w * 100:.0f}% of max)", style=f"bold {watts_style(w, nm.max_w)}")
    parts = nm.node_parts.get(name)
    if parts:
        t.append("  partitions ", style=MUTED)
        for i, p in enumerate(parts):
            ok = nm.part_ok.get(p)
            t.append(("," if i else "") + p, style=GREEN if ok else (RED if ok is False else ""))
        if nm.part_ok and not any(nm.part_ok.get(p) for p in parts):
            t.append(" (not open to your projects)", style=RED)
    job = nm.node_job.get(name)
    if job is not None:
        t.append(f"  job {job.id} ", style="bold")
        t.append(f"{job.user}", style=f"bold {PINK}" if job.user == nm.user else "")
        t.append(f" · {job.account} · {job.name[:30]} · {job.nodes:,} nodes · ", style=MUTED)
        t.append(f"{human_dur(job.remaining)} left", style=AMBER)
    return t


def twin_panel(state, width):
    """The twin's next 24 hours: busy nodes and waiting jobs, when the backlog clears, my next start."""
    tw = state.twin
    now = datetime.now()
    sw = max(width - 24, 10)
    ts = [i * 86400 / sw for i in range(sw)]
    busy = [tw.busy_at(t) for t in ts]
    wait = [tw.waiting_at(t) for t in ts]
    t = Text(f"{tw.simulated:,} eligible jobs through RAPS's scheduler · {rel_time(tw.at, now)}\n", style=MUTED)
    t.append(f"{'busy nodes':<15}", style=MUTED)
    t.append_text(spark(busy, sw, GREEN, lo=0, hi=max(tw.usable, 1)))
    t.append(f" {busy[-1] / max(tw.usable, 1) * 100:5.1f}%\n", style=f"bold {GREEN}")
    t.append(f"{'waiting jobs':<15}", style=MUTED)
    t.append_text(spark(wait, sw, AMBER))
    t.append(f" {human_num(wait[-1]):>6}\n", style=f"bold {AMBER}")
    if tw.power_now is not None:
        power = [tw.power_at(t) / 1e6 for t in ts]
        t.append(f"{'system power':<15}", style=MUTED)
        t.append_text(spark(power, sw, PINK, lo=0, hi=max(power)))
        t.append(f" {power[-1]:6.2f}\n", style=f"bold {PINK}")
    t.append(f"{'':<15}{'now':<{sw // 2}}{'+12 h':<{sw - sw // 2}}+24 h\n", style="#475569")
    if tw.power_now is not None:
        (pk, tpk), (lo, tlo) = peak_low(tw)
        t.append("power ", style=MUTED)
        t.append(f"{tw.power_now / 1e6:.2f} MW now", style=f"bold {PINK}")
        t.append(f" · peak {pk / 1e6:.2f} MW {'now' if tpk == 0 else 'in ' + human_dur(tpk)}", style=PINK)
        t.append(f" ({pk / max(state.machine_max_w, 1) * 100:.0f}% of max)", style=MUTED)
        t.append(f" · low {lo / 1e6:.2f} MW {'now' if tlo == 0 else 'in ' + human_dur(tlo)}\n", style=MUTED)
    starts = [s for s in tw.starts.values() if s is not None]
    if starts:
        t.append("every eligible job started ", style=MUTED)
        t.append(rel_time(max(starts), now), style=f"bold {CYAN}")
        never = sum(1 for s in tw.starts.values() if s is None)
        if never:
            t.append(f" · {never} not within 14 days", style=MUTED)
    mine = sorted((tw.starts[j.id], j) for j in state.my_jobs if j.pending and tw.starts.get(j.id))
    if mine:
        s, j = mine[0]
        t.append("\nyour next job ", style=MUTED)
        t.append(f"{j.id}", style=f"bold {PINK}")
        t.append(" starts ", style=MUTED)
        t.append(rel_time(s, now) if s > now else "now", style=f"bold {CYAN}")
        if j.start:
            t.append(f" (slurm: {rel_time(j.start, now)})", style=MUTED)
    return t


# ---------------------------------------------------------------------------------------------
# 2 · My Jobs
# ---------------------------------------------------------------------------------------------


class MyJobs(LiveView):
    BINDINGS = [
        Binding("enter", "details", "Details"),
        Binding("c", "cancel", "Cancel job"),
        Binding("h", "hold", "Hold"),
        Binding("u", "release", "Release"),
    ]
    DEFAULT_CSS = """
    MyJobs #mj-table { height: 2fr; min-height: 6; }
    MyJobs.empty #mj-table { height: 5; min-height: 5; }
    MyJobs #mj-hist { height: 1fr; }
    MyJobs #mj-lower { height: 3fr; }
    MyJobs #mj-hist { width: 3fr; }
    MyJobs #mj-side { width: 2fr; min-width: 40; }
    MyJobs #mj-side > TextPane { height: auto; }
    MyJobs DataTable { background: $panel; }
    """

    COLS = ("job", "name", "project", "state", "nodes", "progress", "left / limit", "forecast", "W/node", "start",
            "queue", "reason")
    HCOLS = ("job", "name", "project", "state", "nodes", "ran", "of limit", "ended", "exit")

    def compose(self) -> ComposeResult:
        yield _panel(DataTable(id="mj-table", cursor_type="row", zebra_stripes=True), "my active jobs")
        with Horizontal(id="mj-lower"):
            yield _panel(DataTable(id="mj-hist", cursor_type="row", zebra_stripes=True), "recent history")
            with VerticalScroll(id="mj-side"):
                yield _panel(TextPane(_waiting("showusage"), id="mj-alloc"), "allocations · node-hours")
                yield _panel(TextPane(_waiting("sshare"), id="mj-fs"), "fair share")
                yield _panel(TextPane(_waiting("history"), id="mj-stats"), "my last week")

    def on_mount(self):
        self.query_one("#mj-table", DataTable).add_columns(*self.COLS)
        self.query_one("#mj-hist", DataTable).add_columns(*self.HCOLS)

    def render_state(self, state):
        jobs = state.jobs or []
        mine = sorted(state.my_jobs, key=lambda j: (not j.running, -j.priority))
        table = self.query_one("#mj-table", DataTable)
        table.border_subtitle = f"{len(mine)} jobs · enter details · c cancel · h hold · u release"
        now = datetime.now()
        rows = []
        for j in mine:
            if j.running and j.limit:
                frac = j.elapsed / j.limit
                prog = bar(frac, 16, GREEN, heat(frac)) + Text(f" {frac * 100:3.0f}%", style=MUTED)
                left = Text.assemble((human_dur(j.remaining), f"bold {heat(frac)}"), (f" / {human_dur(j.limit)}", MUTED))
                start = Text(rel_time(j.start, now), style=MUTED)
                queue = Text("—", style=MUTED)
            else:
                wait = (now - j.submit).total_seconds() if j.submit else None
                prog = Text(f"waiting {human_dur(wait)}", style=AMBER)
                left = Text(f"limit {human_dur(j.limit)}", style=MUTED)
                start = twin_start(state, j) or Text()
                if start.plain:
                    start = Text.assemble(("twin ", MUTED), start, ("  ", ""))
                start.append(f"slurm {rel_time(j.start, now)}" if j.start else "no slurm estimate", style=MUTED)
                qp = queue_position(j, jobs)
                queue = (Text.assemble((f"#{qp.rank}", f"bold {VIOLET}"), (f"/{qp.of}", MUTED),
                                       (f" · {human_num(qp.ahead_nh)} nh ahead", MUTED))
                         if qp and eligible(j) else Text("blocked", style=SLATE))
            reason = Text(j.reason if j.pending else (j.nodelist[:28] + ("…" if len(j.nodelist) > 28 else "")),
                          style=AMBER if j.pending else MUTED)
            rows.append((Text(j.id, style="bold"), j.name[:28], j.account, state_text(j.state),
                         Text(f"{j.nodes:,}", justify="right"), prog, left, forecast_text(state, j, long=True),
                         power_now_text(state, j) if j.running else power_text(state, j), start, queue, reason))
        sync_table(table, [j.id for j in mine], rows)
        self.set_class(not mine, "empty")
        if not mine:
            table.border_subtitle = "nothing in the queue · submit with sbatch from the Shell tab (6)"

        hist = self.query_one("#mj-hist", DataTable)
        if state.history is not None:
            h = state.history[:200]
            hrows = []
            for j in h:
                frac = j.elapsed / j.limit if j.limit else None
                exit_ok = j.exit_code in ("0:0", "")
                hrows.append((j.id, j.name[:26], j.account, state_text(j.state), Text(f"{j.nodes:,}", justify="right"),
                              human_dur(j.elapsed),
                              Text(f"{frac * 100:3.0f}%", style=heat(frac)) if frac is not None else "—",
                              Text(rel_time(j.end, now) if j.end else "running", style=MUTED),
                              Text(j.exit_code, style=GREEN if exit_ok else f"bold {RED}")))
            sync_table(hist, [j.id for j in h], hrows)
            hist.border_subtitle = f"{len(state.history)} jobs in the last {self.app.poller.slurm.history_days} days"
            self.query_one("#mj-stats", TextPane).update(history_stats(state.history))

        if state.allocations is not None:
            t = Text()
            w = max(self.query_one("#mj-side").size.width - 30, 8)
            for i, a in enumerate(state.allocations):
                if i:
                    t.append("\n")
                t.append(f"{a.project:<11}", style="bold")
                if a.allocation:
                    frac = a.usage / a.allocation
                    t.append_text(bar(frac, w, GREEN, heat(min(frac, 1))))
                    t.append(f" {frac * 100:4.0f}%", style=f"bold {heat(min(frac, 1))}")
                    t.append(f" {human_num(a.usage)}/{human_num(a.allocation)}", style=MUTED)
                else:
                    t.append("no allocation", style=MUTED)
                if a.my_usage:
                    t.append(f"\n{'':<11}you: {human_num(a.my_usage)} nh", style=PINK)
            self.query_one("#mj-alloc", TextPane).update(t or Text("no projects", style=MUTED))

        if state.fairshare is not None:
            t = Text()
            for i, f in enumerate(state.fairshare):
                if i:
                    t.append("\n")
                t.append(f"{f.account:<11}", style="bold")
                t.append_text(bar(f.fairshare, 16, RED, GREEN))
                t.append(f" {f.fairshare:.3f}", style=f"bold {heat(1 - f.fairshare)}")
            self.query_one("#mj-fs", TextPane).update(t)

    def _selected(self):
        table = self.query_one("#mj-table", DataTable)
        if not table.row_count:
            return None
        key = table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
        return next((j for j in (self.state.my_jobs if self.state else []) if j.id == key), None)

    def action_details(self):
        j = self._selected()
        if j:
            self.app.show_job(j.id)

    def _act(self, verb, cmd):
        j = self._selected()
        if j is None:
            self.app.notify("no job selected", severity="warning")
            return
        self.app.confirm_and_run(
            Text.assemble((f"{verb} job ", "bold"), (j.id, f"bold {CYAN}"), (f" ({j.name}, {j.nodes} nodes)?\n\n", "bold"),
                          (" ".join(cmd + [j.id]), "dim")),
            cmd + [j.id])

    def action_cancel(self):
        self._act("Cancel", ["scancel"])

    def action_hold(self):
        self._act("Hold", ["scontrol", "hold"])

    def action_release(self):
        self._act("Release", ["scontrol", "release"])

    @on(DataTable.RowSelected, "#mj-table")
    def _picked(self, event):
        self.app.show_job(event.row_key.value)

    @on(DataTable.RowSelected, "#mj-hist")
    def _hpicked(self, event):
        self.app.show_job(event.row_key.value)


def history_stats(history):
    from collections import Counter
    c = Counter(j.state for j in history)
    nh = sum(j.nodes * j.elapsed for j in history) / 3600
    t = Text()
    t.append(f"{len(history)} jobs", style="bold")
    t.append(f" · {human_num(nh)} node-hours used\n", style=MUTED)
    for st, n in c.most_common():
        t.append_text(state_text(st))
        t.append(f" {n}   ")
    return t


# ---------------------------------------------------------------------------------------------
# 3 · Queue
# ---------------------------------------------------------------------------------------------

SORTS = [
    ("priority", lambda j: (-j.priority,)),
    ("nodes", lambda j: (-j.nodes,)),
    ("submitted", lambda j: (j.submit or datetime.max,)),
    ("user", lambda j: (j.user, -j.priority)),
    ("node-hours", lambda j: (-j.node_hours,)),
    ("energy", None),  # biggest estimated energy first; needs the forecasts, so Queue.render_state builds the key
]
FILTERS = ["all", "running", "pending", "eligible"]


class Queue(LiveView):
    BINDINGS = [
        Binding("slash", "filter", "Filter"),
        Binding("escape", "clear", "Clear", show=False),
        Binding("f", "cycle_filter", "Running/Pending"),
        Binding("m", "mine", "Only mine"),
        Binding("s", "sort", "Sort"),
        Binding("e", "sort_energy", "Sort by energy"),
    ]
    DEFAULT_CSS = """
    Queue #q-head { height: 1; padding: 0 1; }
    Queue #q-filter { display: none; height: 3; }
    Queue.filtering #q-filter { display: block; }
    Queue #q-table { height: 1fr; background: $panel; }
    """
    COLS = ("job", "user", "project", "name", "state", "nodes", "limit", "forecast", "W/node", "W now", "energy",
            "ran / waited",
            "twin start", "slurm start", "node-h", "reason", "priority", "qos")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.text = ""
        self.only_mine = False
        self.filt = 0
        self.sort = 0

    def compose(self) -> ComposeResult:
        yield TextPane("", id="q-head")
        yield Input(placeholder="filter: user, project, name, job id, reason, partition …", id="q-filter")
        yield _panel(DataTable(id="q-table", cursor_type="row", zebra_stripes=True), "queue")

    def on_mount(self):
        self.query_one("#q-table", DataTable).add_columns(*self.COLS)

    def render_state(self, state):
        if state.jobs is None:
            return
        now = datetime.now()
        jobs = state.jobs
        f = FILTERS[self.filt]
        if f == "running":
            jobs = [j for j in jobs if j.running]
        elif f == "pending":
            jobs = [j for j in jobs if j.pending]
        elif f == "eligible":
            jobs = [j for j in jobs if eligible(j)]
        if self.only_mine:
            jobs = [j for j in jobs if j.user == state.user]
        if self.text:
            needle = self.text.lower()
            jobs = [j for j in jobs if needle in " ".join((j.id, j.user, j.account, j.name, j.reason, j.partition,
                                                           j.qos, j.state)).lower()]
        energy = {j.id: job_energy(state, j) for j in jobs}
        sort_key = SORTS[self.sort][1] or (lambda j: (-(energy[j.id][0] if energy[j.id] else -1.0),))
        jobs = sorted(jobs, key=lambda j: (not j.running,) + sort_key(j))

        rows = []
        for j in jobs:
            mine = j.user == state.user
            if j.running:
                ran = Text(human_dur(j.elapsed), style=GREEN)
                start = Text(rel_time(j.start, now), style=MUTED)
            else:
                ran = Text(human_dur((now - j.submit).total_seconds()) if j.submit else "—", style=AMBER)
                start = Text(rel_time(j.start, now), style=CYAN) if j.start else Text("—", style=MUTED)
            rows.append((
                Text(j.id, style=f"bold {PINK}" if mine else "bold"),
                Text(j.user, style=f"bold {PINK}" if mine else ""),
                j.account, j.name[:30], state_text(j.state),
                Text(f"{j.nodes:,}", justify="right", style=f"bold {heat(min(j.nodes / 9408, 1) ** 0.4)}"),
                human_dur(j.limit), forecast_text(state, j), power_text(state, j), power_now_text(state, j), energy_text(energy[j.id]),
                ran,
                twin_start(state, j) or Text("—", style=SLATE), start,
                Text(human_num(j.node_hours), justify="right"),
                Text(j.reason[:26], style=AMBER if j.pending else MUTED),
                Text(f"{j.priority:,}", justify="right", style=MUTED), j.qos,
            ))
        table = self.query_one("#q-table", DataTable)
        sync_table(table, [j.id for j in jobs], rows)
        total = sum(e[0] for e in energy.values() if e)
        table.border_title = (f"queue · {len(jobs):,} of {len(state.jobs):,} jobs · "
                              f"est. {total / 1000:,.1f} MWh over their runs")
        table.border_subtitle = ("forecast ● same job ◐ same user ○ same project · "
                                 "enter details · / filter · f state · m mine · s sort")

        head = Text()
        for label, on_ in ((f"showing {f}", True), ("only mine", self.only_mine)):
            head.append(f" {label} ", style=f"bold #0b0f17 on {CYAN}" if on_ else f"{MUTED} on #1e293b")
            head.append(" ")
        head.append(f" sort: {SORTS[self.sort][0]} ", style=f"bold #0b0f17 on {VIOLET}")
        if self.text:
            head.append(f"  filter “{self.text}”", style=AMBER)
        self.query_one("#q-head", TextPane).update(head)

    def _rerender(self):
        if self.state is not None:
            self.render_state(self.state)

    def action_filter(self):
        self.add_class("filtering")
        self.query_one("#q-filter", Input).focus()

    def action_clear(self):
        if self.app.ff is not None and not self.text and not self.has_class("filtering"):
            self.app.action_ff_stop()   # nothing to clear: esc leaves fast-forward
            return
        inp = self.query_one("#q-filter", Input)
        inp.value = ""
        self.text = ""
        self.remove_class("filtering")
        self.query_one("#q-table").focus()
        self._rerender()

    @on(Input.Changed, "#q-filter")
    def _typed(self, event):
        self.text = event.value.strip()
        self._rerender()

    @on(Input.Submitted, "#q-filter")
    def _done(self, event):
        self.query_one("#q-table").focus()
        if not self.text:
            self.remove_class("filtering")

    def action_cycle_filter(self):
        if self.app.ff_key():   # `ff`: the first f already cycled the filter, undo that
            self.filt = (self.filt - 1) % len(FILTERS)
            self._rerender()
            return
        self.filt = (self.filt + 1) % len(FILTERS)
        self._rerender()

    def action_mine(self):
        self.only_mine = not self.only_mine
        self._rerender()

    def action_sort_energy(self):
        self.sort = [name for name, _ in SORTS].index("energy")
        self._rerender()

    def action_sort(self):
        self.sort = (self.sort + 1) % len(SORTS)
        self._rerender()

    @on(DataTable.RowSelected, "#q-table")
    def _picked(self, event):
        self.app.show_job(event.row_key.value)


# ---------------------------------------------------------------------------------------------
# 4 · Nodes
# ---------------------------------------------------------------------------------------------


class Nodes(LiveView):
    BINDINGS = [Binding("c", "color_mode", "Color mode")]
    DEFAULT_CSS = """
    Nodes #nd-head { height: 1; padding: 0 1; }
    Nodes #nd-box { height: 1fr; }
    Nodes #nd-map { height: 1fr; }
    Nodes #nd-info { height: 2; }
    """

    def compose(self) -> ComposeResult:
        yield TextPane("", id="nd-head")
        with Vertical(id="nd-box") as box:
            from raps.live import widgets
            _panel(box, f"{self.app.poller.system or 'cluster'} · one pixel per node · "
                        f"{widgets.NODES_PER_CAB} nodes per cabinet")
            yield NodeMap(mode="power", id="nd-map")  # the job states are on the Overview
            yield TextPane(_waiting("sinfo -N"), id="nd-info")

    def render_state(self, state):
        nm = self.query_one("#nd-map", NodeMap)
        if state.node_states:
            nm.set_data(state.node_states, node_job_map(state.jobs or []), state.user, node_power=state.node_power,
                        max_w=state.node_max_w, **node_access(state))
            hovered = getattr(self, "hovered", None)
            self.query_one("#nd-info", TextPane).update(map_footer(nm, hovered, NODES_HINT))
        head = Text()
        for m in NodeMap.MODES:
            sel = m == nm.mode
            head.append(f" {m} ", style=f"bold #0b0f17 on {CYAN}" if sel else f"{MUTED} on #1e293b")
            head.append(" ")
        head.append("  c colors · hover or arrow keys for a node · click or enter for its job", style=MUTED)
        upd = state.updated.get("nodes")
        if upd:
            head.append(f"   node states {rel_time(upd)}", style="dim")
        self.query_one("#nd-head", TextPane).update(head)

    def action_color_mode(self):
        nm = self.query_one("#nd-map", NodeMap)
        nm.mode = NodeMap.MODES[(NodeMap.MODES.index(nm.mode) + 1) % len(NodeMap.MODES)]
        if self.state is not None:
            self.render_state(self.state)

    @on(NodeMap.Hover)
    def _hover(self, event):
        self.hovered = event.nodes
        nm = self.query_one("#nd-map", NodeMap)
        self.query_one("#nd-info", TextPane).update(map_footer(nm, event.nodes, NODES_HINT))

    @on(NodeMap.Clicked)
    def _click(self, event):
        self.app.open_node_job(event.nodes)


# ---------------------------------------------------------------------------------------------
# 5 · System
# ---------------------------------------------------------------------------------------------


class System(LiveView):
    DEFAULT_CSS = """
    System #sy-top { height: auto; max-height: 50%; }
    System #sy-top > * { height: auto; }
    System #sy-bottom { height: 1fr; }
    System #sy-res { width: 3fr; background: $panel; }
    System #sy-reasons, System #sy-tops { width: 2fr; }
    System #sy-users, System #sy-projects { height: 1fr; }
    System #sy-parts { width: 3fr; background: $panel; }
    """

    def compose(self) -> ComposeResult:
        with Horizontal(id="sy-top"):
            yield _panel(DataTable(id="sy-parts", zebra_stripes=True, cursor_type="none"), "partitions")
            yield _panel(TextPane(_waiting("squeue"), id="sy-reasons"), "why jobs are waiting")
        with Horizontal(id="sy-bottom"):
            yield _panel(DataTable(id="sy-res", zebra_stripes=True, cursor_type="row"), "reservations")
            with Vertical(id="sy-tops"):
                yield _panel(TextPane(_waiting("squeue"), id="sy-users"), "top users · running nodes")
                yield _panel(TextPane(_waiting("squeue"), id="sy-projects"), "top projects · running nodes")

    def on_mount(self):
        self.query_one("#sy-parts", DataTable).add_columns("partition", "state", "nodes", "alloc", "idle", "other",
                                                          "busy", "max time")
        self.query_one("#sy-res", DataTable).add_columns("reservation", "state", "nodes", "starts", "ends", "flags")

    def render_state(self, state):
        now = datetime.now()
        if state.partitions:
            rows = [(Text(p.name + (" *" if p.default else ""), style="bold"),
                     Text(p.avail, style=GREEN if p.avail == "up" else RED), f"{p.total:,}", f"{p.alloc:,}",
                     f"{p.idle:,}", f"{p.other:,}",
                     bar(p.alloc / p.total if p.total else 0, 14, GREEN, CYAN), p.timelimit)
                    for p in state.partitions]
            sync_table(self.query_one("#sy-parts", DataTable), [p.name for p in state.partitions], rows)
        if state.reservations is not None:
            res = sorted(state.reservations, key=lambda r: (r.state != "ACTIVE", -r.nodes))
            rows = []
            for r in res:
                big = r.nodes >= 1000 or "MAINT" in r.flags and r.nodes >= 100
                rows.append((Text(r.name[:40], style=f"bold {RED}" if big else "bold"),
                             Text(r.state.lower(), style=GREEN if r.state == "ACTIVE" else AMBER),
                             Text(f"{r.nodes:,}", justify="right"), Text(rel_time(r.start, now), style=MUTED),
                             Text(rel_time(r.end, now), style=MUTED), Text(r.flags[:36], style=MUTED)))
            sync_table(self.query_one("#sy-res", DataTable), [r.name for r in res], rows)
        if state.jobs is not None:
            w = max(self.query_one("#sy-reasons").size.width - 4, 30)
            rc = reason_counts(state.jobs)[:14]
            peak = max((n for _, n in rc), default=1)
            t = Text()
            for i, (reason, n) in enumerate(rc):
                if i:
                    t.append("\n")
                color = AMBER if reason in ("Priority", "Resources") else VIOLET if "Depend" in reason else SLATE
                t.append(f"{reason[:22]:<23}", style="bold")
                t.append_text(bar(n / peak, max(w - 31, 6), color, color))
                t.append(f" {n:>5,}", style=color)
            self.query_one("#sy-reasons", TextPane).update(t)

            self._top_list("#sy-users", state, "user")
            self._top_list("#sy-projects", state, "account")

    def _top_list(self, pane, state, key):
        """Bars of running nodes per user or per project (Slurm account, e.g. cfd227), biggest first."""
        rows = top_accounts(state.jobs, n=max(self.query_one(pane).size.height - 2, 3), key=key)
        peak = max((r[1] for r in rows), default=1) or 1
        w = max(self.query_one(pane).size.width - 4, 30)
        t = Text()
        for i, (name, rn, rj, pnh) in enumerate(rows):
            if i:
                t.append("\n")
            mine = name == state.user if key == "user" else name in (state.my_accounts or ())
            t.append(f"{name[:12]:<13}", style=f"bold {PINK}" if mine else "bold")
            t.append_text(bar(rn / peak, max(w - 34, 6), CYAN, VIOLET))
            t.append(f" {rn:>5,}n {rj:>3}j", style=MUTED)
            t.append(f" {human_num(pnh):>6}nh", style=AMBER)
        self.query_one(pane, TextPane).update(t)


# ---------------------------------------------------------------------------------------------
# 6 · Shell
# ---------------------------------------------------------------------------------------------

QUICK = [
    "squeue --me", "squeue --me --start", "sacct -X --starttime now-1day", "showusage", "sshare -U",
    "sinfo -s", "scontrol show res", "sbatch ", "salloc -A  -N 1 -t 30 ", "module list",
]


TAB_DIGITS = {"1", "2", "3", "4", "5", "7"}  # typed alone in the shell and Enter: go to that tab (6 is the shell)
BASH_NOISE = ("bash: cannot set terminal process group", "bash: no job control in this shell",
              "Shell cwd was reset to")
EDITORS = {"vi", "vim", "nvim", "nano", "pico", "micro", "emacs", "edit", "e"}   # open in the built-in editor
VIEWERS = {"view", "less", "more"}                                                   # same, read-only
NEEDS_TERMINAL = {"top", "htop", "btop", "watch", "ssh", "tmux", "screen", "mc", "ncdu", "info"}
ENV_SKIP = ("_", "SHLVL", "OLDPWD", "PWD", "COLUMNS", "LINES")


class ShellInput(Input):
    """The shell prompt. A digit typed into an empty prompt jumps to that tab (6 is the shell itself)."""

    def on_key(self, event):
        if not self.value and event.character in TAB_DIGITS:
            event.stop()
            event.prevent_default()  # keeps Input from inserting the digit
            self.app.action_tab(int(event.character) - 1)


class Shell(LiveView):
    """
    A bash prompt: every command runs in an interactive bash (so ~/.bashrc aliases work) on a pty (so
    ls prints columns and colors). The working directory and exported variables carry over from one
    command to the next. Tab completes commands and paths; 1-5 or 7 on an empty prompt switches tab.
    """

    BINDINGS = [Binding("ctrl+k,ctrl+c", "kill", "Kill command", priority=True), Binding("escape", "leave", "Leave prompt"),
                Binding("ctrl+l", "clear", "Clear")]
    DEFAULT_CSS = """
    Shell #sh-log { height: 1fr; background: $panel; border: round $primary-darken-2;
                    border-title-color: $secondary; border-title-style: bold; padding: 0 1; }
    Shell #sh-input { dock: bottom; border: tall $accent; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cwd = Path.cwd()
        self.oldcwd = self.cwd
        self.env = dict(os.environ)
        self._path_cmds = None
        self.proc = None
        self.hist_path = cache_dir() / "shell_history"
        try:
            self.history = self.hist_path.read_text().splitlines()[-500:]
        except OSError:
            self.history = []
        self.hpos = None

    def compose(self) -> ComposeResult:
        log = RichLog(id="sh-log", wrap=True, markup=False, highlight=False, max_lines=5000)
        log.border_title = "shell"
        yield log
        yield ShellInput(placeholder="command", id="sh-input",
                    suggester=SuggestFromList(list(dict.fromkeys(reversed(self.history))) + QUICK,
                                              case_sensitive=True))

    def on_mount(self):
        log = self.query_one("#sh-log", RichLog)
        log.write(gradient(f"◆ {self.app.poller.system or 'cluster'} shell"))
        log.write(Text("try: " + " · ".join(QUICK[:6]), style=MUTED))
        self._prompt_title()

    def on_show(self):
        super().on_show()
        self.query_one("#sh-input", Input).focus()

    def _prompt_title(self):
        home = str(Path.home())
        cwd = str(self.cwd)
        self.query_one("#sh-log", RichLog).border_title = "shell · " + ("~" + cwd[len(home):] if cwd.startswith(home) else cwd)

    @on(Input.Submitted, "#sh-input")
    def _run(self, event):
        cmd = event.value.strip()
        event.input.value = ""
        self.hpos = None
        if not cmd:
            return
        if cmd in TAB_DIGITS:
            self.app.action_tab(int(cmd) - 1)
            return
        if cmd == "!!" and self.history:
            cmd = self.history[-1]
        if not self.history or self.history[-1] != cmd:
            self.history.append(cmd)
            try:
                self.hist_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.hist_path, "a") as f:
                    f.write(cmd + "\n")
            except OSError:
                pass
        log = self.query_one("#sh-log", RichLog)
        if cmd == "clear":
            log.clear()
            return
        log.write(Text.assemble(("\n❯ ", f"bold {PINK}"), (cmd, "bold")))
        if cmd == "cd" or (cmd.startswith("cd ") and not re.search(r"[;&|<>$`(]", cmd)):
            arg = cmd[3:].strip() or "~"
            target = str(self.oldcwd) if arg == "-" else os.path.expanduser(os.path.expandvars(arg))
            new = (self.cwd / target).resolve()
            if new.is_dir():
                self.oldcwd, self.cwd = self.cwd, new
                self._prompt_title()
            else:
                log.write(Text(f"cd: no such directory: {target}", style=RED))
            return
        try:
            words = shlex.split(cmd)
        except ValueError:  # unbalanced quote: bash will complain
            words = []
        prog = os.path.basename(words[0]) if words else ""
        if prog in EDITORS | VIEWERS:
            files = [w for w in words[1:] if not w.startswith("-")]
            if len(files) != 1:
                log.write(Text(f"{prog}: give one file name (this opens the built-in editor)", style=AMBER))
                return
            from raps.live.screens import EditScreen
            self.app.push_screen(EditScreen(self.cwd / os.path.expanduser(files[0]), read_only=prog in VIEWERS))
            return
        if prog in NEEDS_TERMINAL or (prog in ("python", "python3", "ipython", "R", "julia") and len(words) == 1):
            log.write(Text(f"{prog} needs a full terminal and cannot run inside the console "
                           f"(quit with ctrl+q, or use another terminal)", style=AMBER))
            return
        if self.proc is not None:
            log.write(Text("a command is still running (ctrl+k to kill it)", style=AMBER))
            return
        threading.Thread(target=self._exec, args=(cmd,), daemon=True).start()

    def _exec(self, cmd):
        log = self.query_one("#sh-log", RichLog)
        write = lambda t: self.app.call_from_thread(log.write, t)  # noqa: E731
        t0 = datetime.now()
        fd, state = tempfile.mkstemp(prefix="raps-shell-")
        os.close(fd)
        rc = None
        master = slave = None
        try:
            master, slave = pty.openpty()
            cols = max(log.size.width - 4, 80)
            fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 50, cols, 0, 0))
            env = {**self.env, "COLUMNS": str(cols), "TERM": "xterm-256color", "PAGER": "cat", "GIT_PAGER": "cat"}
            # after the command, record where bash ended up and which variables are exported
            script = (f"{cmd}\n__rc=$?\npwd -P > {shlex.quote(state)}.cwd\nenv -0 > {shlex.quote(state)}.env\n"
                      f"exit $__rc")
            self.proc = subprocess.Popen(["bash", "-i", "-c", script], cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
                                         stdout=slave, stderr=slave, start_new_session=True)
            os.close(slave)
            slave = None
            held = None  # the last line, held back: an interactive bash ends its output with "exit"
            for line in self._lines(master):
                if line.startswith(BASH_NOISE):
                    continue
                if held is not None:
                    write(Text.from_ansi(held))
                held = line
            if held is not None and held.strip() != "exit":
                write(Text.from_ansi(held))
            rc = self.proc.wait()
            self._absorb(state)
        except Exception as exc:
            write(Text(str(exc), style=RED))
        finally:
            self.proc = None
            for fd in (master, slave):
                if fd is not None:
                    os.close(fd)
            for ext in ("", ".cwd", ".env"):
                try:
                    os.unlink(state + ext)
                except OSError:
                    pass
        dt = (datetime.now() - t0).total_seconds()
        style = GREEN if rc == 0 else RED
        write(Text(f"└─ exit {rc} · {dt:.1f}s", style=f"dim {style}"))
        self.app.call_from_thread(self._prompt_title)
        # anything Slurm-shaped probably changed the queue
        if any(w in cmd for w in ("sbatch", "scancel", "scontrol", "salloc")):
            self.app.call_from_thread(self.app.poller.refresh, "jobs", "history")

    @staticmethod
    def _lines(master):
        """Lines of what the pty produces. A carriage return redraws the line (progress bars), so keep its last part."""
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        buf = ""
        while True:
            try:
                data = os.read(master, 65536)
            except OSError:  # EIO: the last process holding the pty has gone
                break
            if not data:
                break
            buf += dec.decode(data)
            *done, buf = buf.split("\n")
            for line in done:
                yield line.rstrip("\r").rsplit("\r", 1)[-1]
        if buf.strip():
            yield buf.rsplit("\r", 1)[-1]

    def _absorb(self, state):
        """Pick up the working directory and exported variables the command ended with (absent after `exit`)."""
        try:
            new = Path(Path(state + ".cwd").read_text().strip())
            if new.is_dir() and new != self.cwd:
                self.oldcwd, self.cwd = self.cwd, new
            for item in Path(state + ".env").read_bytes().decode(errors="replace").split("\0"):
                k, eq, v = item.partition("=")
                if eq and k not in ENV_SKIP and not k.startswith("BASH_FUNC_"):
                    self.env[k] = v
        except OSError:
            pass

    def action_leave(self):
        self.app.set_focus(None)

    def action_kill(self):
        log = self.query_one("#sh-log", RichLog)
        proc = self.proc  # the worker thread clears self.proc when the command ends
        if proc is None:
            log.write(Text("nothing is running", style=MUTED))
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # bash and everything it started: same process group
        except OSError:
            proc.kill()
        log.write(Text("killed", style=RED))

    def action_clear(self):
        self.query_one("#sh-log", RichLog).clear()

    def _commands(self):
        if self._path_cmds is None:
            names = set()
            for d in self.env.get("PATH", "").split(":"):
                try:
                    names.update(e for e in os.listdir(d) if os.access(os.path.join(d, e), os.X_OK))
                except OSError:
                    pass
            self._path_cmds = sorted(names)
        return self._path_cmds

    def complete(self, inp):
        """Tab: complete the word before the cursor as a command (first word) or a path; list ties."""
        text = inp.value[:inp.cursor_position]
        head, _, word = text.rpartition(" ")
        dirpart, _, base = word.rpartition("/")
        dirpart = dirpart + "/" if "/" in word else ""
        folder = Path(os.path.expanduser(dirpart)) if dirpart else Path()
        folder = folder if folder.is_absolute() else self.cwd / folder
        try:
            names = [n + ("/" if (folder / n).is_dir() else "") for n in sorted(os.listdir(folder))
                     if n.startswith(base) and (base.startswith(".") or not n.startswith("."))]
        except OSError:
            names = []
        found = [dirpart + n for n in names]
        if not head.strip() and not dirpart:
            found = sorted(set(found) | {c for c in self._commands() if c.startswith(word)})
        if not found:
            return
        common = os.path.commonprefix(found)
        if len(found) > 1:
            self.query_one("#sh-log", RichLog).write(Text("  ".join(os.path.basename(f.rstrip("/")) + ("/" if f.endswith("/") else "") for f in found[:60]), style=MUTED))
        if len(common) > len(word) or len(found) == 1:
            done = common if len(found) > 1 or common.endswith("/") else common + " "
            inp.value = head + (" " if head or text.startswith(" ") else "") + done + inp.value[inp.cursor_position:]
            inp.cursor_position = len(head) + (1 if head or text.startswith(" ") else 0) + len(done)

    def on_key(self, event):
        inp = self.query_one("#sh-input", Input)
        if not inp.has_focus or event.key not in ("up", "down") or not self.history:
            return
        if event.key == "up":
            self.hpos = len(self.history) - 1 if self.hpos is None else max(self.hpos - 1, 0)
        else:
            if self.hpos is None:
                return
            self.hpos += 1
            if self.hpos >= len(self.history):
                self.hpos = None
                inp.value = ""
                event.stop()
                return
        inp.value = self.history[self.hpos]
        inp.cursor_position = len(inp.value)
        event.stop()



# ---------------------------------------------------------------------------------------------
# 7 · Launch: where and when could a job of mine run?
# ---------------------------------------------------------------------------------------------


def _short(msg, n=44):
    """Slurm filter messages, cut to their first clause ('You are not a member of partition g1')."""
    msg = msg.split(". ")[0]
    return msg if len(msg) <= n else msg[:n - 1] + "…"


class Launch(LiveView):
    """
    Where you may submit (sbatch --test-only per project x partition), what would start right now
    (the twin's plan at time limits), and Slurm's and the twin's start for a job you describe.
    Enter on a row puts its sbatch line into the Shell tab.
    """

    BINDINGS = [Binding("a", "next_account", "Project"), Binding("t", "test", "Ask Slurm")]
    DEFAULT_CSS = """
    Launch #ln-form { height: 3; }
    Launch #ln-form Input { width: 18; }
    Launch #ln-form TextPane { width: 1fr; padding: 1 1 0 1; }
    Launch #ln-table { height: auto; max-height: 50%; background: $panel; }
    Launch #ln-lower { height: 1fr; }
    Launch #ln-lower > * { width: 1fr; }
    Launch #ln-lower TextPane { height: auto; }
    """
    COLS = ("partition", "you", "qos", "limits", "idle now", "starts at once?", "slurm start", "twin start", "command")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.account = None
        self.checked = {}      # (account, partition, nodes, seconds) -> (slurm TestResult, twin start)
        self.busy = False

    def compose(self) -> ComposeResult:
        with Horizontal(id="ln-form"):
            yield Input("1", placeholder="nodes", id="ln-nodes")
            yield Input("1h", placeholder="time: 90m, 2h, 1:30:00", id="ln-time")
            yield TextPane("", id="ln-acct")
        yield _panel(DataTable(id="ln-table", cursor_type="row", zebra_stripes=True), "where could it run?")
        with Horizontal(id="ln-lower"):
            with VerticalScroll():
                yield _panel(TextPane(_waiting("twin"), id="ln-now"), "starts at once · nodes → longest time limit")
            with VerticalScroll():
                yield _panel(TextPane(_waiting("sbatch --test-only"), id="ln-access"), "access · per project")

    def on_mount(self):
        self.query_one("#ln-table", DataTable).add_columns(*self.COLS)

    # -- request -------------------------------------------------------------------------------
    def request(self):
        from raps.live.launch import parse_walltime
        try:
            nodes = max(int(self.query_one("#ln-nodes", Input).value.strip()), 1)
        except ValueError:
            nodes = 1
        seconds = parse_walltime(self.query_one("#ln-time", Input).value) or 3600
        return nodes, seconds

    def accounts(self, state):
        acc = state.access
        return sorted({a for a, _ in acc.results}) if acc is not None else []

    def render_state(self, state):
        from raps.live.launch import qos_for, sbatch_line, start_now_windows
        accounts = self.accounts(state)
        if accounts and self.account not in accounts:
            # Slurm's default account, else (Frontier has none) the project with the most allocation left
            left = {a.project: a.remaining or 0 for a in state.allocations or []}
            default = state.access.default_account
            self.account = default if default in accounts else max(accounts, key=lambda a: left.get(a, -1))
        nodes, seconds = self.request()
        head = Text()
        head.append("project ", style=MUTED)
        head.append(self.account or "…", style=f"bold {PINK}")
        head.append("  (a: next)  ", style=MUTED)
        head.append(f"{nodes:,} nodes × {human_dur(seconds)}", style="bold")
        head.append("   t or enter in a field: ask Slurm (sbatch --test-only, nothing is submitted)",
                    style=MUTED)
        if self.busy:
            head.append("  asking…", style=CYAN)
        self.query_one("#ln-acct", TextPane).update(head)

        layout = state.layout or {}
        acc = state.access
        windows = start_now_windows(state.twin.plan, layout, sizes=(nodes,)) \
            if state.twin is not None and state.twin.plan is not None else {}
        idle = {}
        for name, part in layout.items():
            idle[name] = sum(1 for n in part.nodes
                             if (state.node_states or {}).get(n, "").rstrip("*$~#!%@^-+").lower() == "idle")
        rows, keys = [], []
        for name, part in sorted(layout.items(), key=lambda kv: (not (acc and acc.ok(kv[0], self.account)), kv[0])):
            ok = acc.ok(name, self.account) if acc is not None else None
            you = Text("…", style=SLATE) if ok is None else (Text("✓ yes", style=f"bold {GREEN}") if ok
                                                             else Text("✗ no", style=f"bold {RED}"))
            limits = []
            if part.max_nodes:
                limits.append(f"≤{part.max_nodes:,} n")
            if part.max_wall:
                limits.append(f"≤{human_dur(part.max_wall)}")
            w = dict(windows.get(name, [])).get(nodes, "?")
            if part.max_nodes and nodes > part.max_nodes:
                at_once = Text(f"over the {part.max_nodes}-node limit", style=RED)
            elif w == "?":
                at_once = Text("…", style=SLATE)
            elif w is None:
                at_once = Text("no: not enough free nodes", style=AMBER)
            elif w >= seconds:
                up_to = human_dur(w) if w != float("inf") else "any time"
                at_once = Text(f"yes · up to {up_to}", style=f"bold {GREEN}")
            else:
                at_once = Text(f"only if ≤ {human_dur(w)}", style=AMBER)
            slurm = twin = Text("—", style=SLATE)
            got = self.checked.get((self.account, name, nodes, seconds))
            if got is not None:
                res, tstart = got
                now = datetime.now()
                if res.ok:
                    when = rel_time(res.start, now) if res.start and res.start > now else "now"
                    slurm = Text(when, style=f"bold {CYAN}")
                else:
                    slurm = Text(_short(res.message) or "rejected", style=RED)
                if tstart is not None:
                    twin = Text(rel_time(tstart, now) if tstart > now + timedelta(minutes=1) else "now",
                                style=f"bold {CYAN}")
            elif ok is False:
                slurm = Text(_short(acc.reason(name)) or "rejected", style=RED)
            cmd = sbatch_line(self.account or "<project>", name, nodes, min(seconds, part.max_wall or seconds),
                              qos_for(part)) if ok else ""
            rows.append((Text(name, style="bold"), you, part.job_qos or "normal", " ".join(limits) or "—",
                         Text(f"{idle[name]:,}/{len(part.nodes):,}", justify="right"), at_once, slurm, twin,
                         Text(cmd, style=MUTED)))
            keys.append(name)
        table = self.query_one("#ln-table", DataTable)
        sync_table(table, keys, rows)
        table.border_subtitle = "enter: put the sbatch line in the Shell tab · ✗ = Slurm rejects this project there"

        if state.twin is not None and state.twin.plan is not None and layout:
            sizes = (1, 8, 32, 128, 512, 1024, 2048, 4096)
            allw = start_now_windows(state.twin.plan, layout, sizes=sizes)
            t = Text(f"{'':<13}" + "".join(f"{n:>8,}" for n in sizes) + "\n", style=MUTED)
            for name in keys:
                if acc is not None and not acc.ok(name, self.account):
                    continue
                t.append(f"{name:<13}", style="bold")
                for n, w in allw.get(name, []):
                    if w is None:
                        t.append(f"{'—':>8}", style=SLATE)
                    else:
                        t.append(f"{'any' if w == float('inf') else human_dur(w):>8}", style=f"bold {GREEN}")
                t.append("\n")
            t.append("from the RAPS twin's plan at time limits, which is how Slurm backfills: a job this size\n"
                     "with at most this time limit fits in nodes nobody is planned to use yet.", style="#64748b")
            self.query_one("#ln-now", TextPane).update(t)

        if acc is not None:
            t = Text(f"checked {rel_time(acc.checked)} with a 1-node test job\n", style=MUTED)
            for name in sorted({p for _, p in acc.results}):
                t.append(f"{name:<13}", style="bold")
                for a in accounts:
                    r = acc.results.get((a, name))
                    if r is not None:
                        t.append(f" {a} ", style=f"bold {PINK}" if a == self.account else MUTED)
                        t.append("✓" if r.ok else "✗", style=GREEN if r.ok else RED)
                t.append("\n")
            for name in sorted({p for _, p in acc.results}):
                if not acc.ok(name):
                    t.append(f"\n{name}: ", style=f"bold {RED}")
                    t.append(acc.reason(name)[:200], style=MUTED)
            self.query_one("#ln-access", TextPane).update(t)

    # -- actions -------------------------------------------------------------------------------
    def action_next_account(self):
        accounts = self.accounts(self.state) if self.state else []
        if accounts:
            self.account = accounts[(accounts.index(self.account) + 1) % len(accounts)] \
                if self.account in accounts else accounts[0]
            self.render_state(self.state)

    @on(Input.Changed, "#ln-nodes, #ln-time")
    def _changed(self, event):
        if self.state is not None:
            self.render_state(self.state)

    @on(Input.Submitted, "#ln-nodes, #ln-time")
    def _submitted(self, event):
        self.action_test()

    def action_test(self):
        state = self.state
        if state is None or state.access is None or self.busy:
            return
        nodes, seconds = self.request()
        account = self.account
        parts = [name for name, part in (state.layout or {}).items() if state.access.ok(name, account)]
        slurm = self.app.poller.slurm
        partition_nodes = self.app.poller.twin.partition_nodes if self.app.poller.twin else None
        self.busy = True
        self.render_state(state)

        def work():
            from raps.live.launch import qos_for, what_if
            for name in parts:
                part = state.layout[name]
                secs = min(seconds, part.max_wall or seconds)
                res = slurm.test_only(account, name, nodes, secs, qos=qos_for(part))
                tstart = what_if(state, partition_nodes, name, account, nodes, secs) \
                    if partition_nodes and res.ok else None
                self.checked[(account, name, nodes, seconds)] = (res, tstart)
            self.app.call_from_thread(self._done)
        threading.Thread(target=work, daemon=True, name="raps-live-launch").start()

    def _done(self):
        self.busy = False
        if self.state is not None:
            self.render_state(self.state)

    @on(DataTable.RowSelected, "#ln-table")
    def _picked(self, event):
        cmd = str(self.query_one("#ln-table", DataTable).get_row(event.row_key)[-1])
        if not cmd:
            self.app.notify("no access to that partition with this project", severity="warning")
            return
        self.app.open_shell_with(cmd)


VIEWS = [("overview", "Overview", Overview), ("myjobs", "My Jobs", MyJobs), ("queue", "Queue", Queue),
         ("nodes", "Nodes", Nodes), ("system", "System", System), ("shell", "Shell", Shell),
         ("launch", "Launch", Launch)]

