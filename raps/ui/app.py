"""
RapsApp: the multi-view Textual console.

The simulation runs in a worker thread. After each simulated second it checks a wall-clock gate
and, when due, builds a ``UISnapshot`` and stores it in a single "latest" slot; the UI thread polls
that slot. The simulation therefore never waits on rendering, and a slow terminal just skips
intermediate snapshots.
"""
import threading
import time
import traceback

from rich.cells import cell_len
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import ContentSwitcher, Footer, Static

from raps.ui.widgets.textpane import TextPane

from raps.ui.job_detail import JobDetailScreen
from raps.ui.views import VIEWS

HELP_TEXT = """\
[b]RAPS console[/b]

[b]Simulation[/b]
  space / k    pause or resume
  l / +        faster (coarser time step, then unthrottled)
  j / -        slower (finer time step, then wall-clock throttle)

[b]Views[/b]
  1-6          Overview, Jobs, Node Map, Power, Cooling, Network
  tab / S-tab  next / previous view (only views with data are shown)
  a            auto-cycle views (period set with --ui-cycle SECONDS)

[b]Jobs[/b]      / filter   s sort column   n sort by nodes   r reverse   esc clear filter
           enter or click a job for details (m there highlights it in the node map)
[b]Node Map[/b]  c color mode (state, job, power)   arrows move   enter zoom into a rack   esc back/unhighlight
[b]Cooling[/b]   m next metric   M previous metric

  ?  this help      q  quit (the final report prints after the app exits)

Press escape or ? to close.
"""


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,question_mark,q", "app.pop_screen", "Close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > Static { width: 78; max-width: 100%; height: auto; max-height: 100%;
                          border: round $primary; background: $surface; padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        yield Static(HELP_TEXT)


class RapsApp(App):
    TITLE = "RAPS"
    CSS = """
    #topbar { height: 1; background: $primary-darken-2; color: $text; padding: 0 1; }
    #statusbar { height: 1; padding: 0 1; }
    #views { height: 1fr; }
    """
    BINDINGS = [
        Binding("space,k", "pause", "Pause"),
        Binding("l,plus", "faster", "Faster"),
        Binding("j,minus,underscore", "slower", "Slower"),
        Binding("tab", "next_view", "Next view", show=False, priority=True),
        Binding("shift+tab", "prev_view", "Prev view", show=False, priority=True),
        Binding("a", "toggle_cycle", "Auto-cycle"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ] + [Binding(str(i + 1), f"view_index({i})", v.title, show=False) for i, v in enumerate(VIEWS)]

    def __init__(self, engine=None, builder=None, *, source=None, hz=4.0, cycle=None, autoshutdown=True):
        """
        `engine`/`builder` run a real simulation. Tests can instead pass `source`, an iterable of
        UISnapshots, which the worker feeds exactly as it would feed built snapshots.
        """
        super().__init__()
        self.engine = engine
        self.builder = builder
        self.source = source
        self.hz = hz
        self.cycle = cycle
        self.autoshutdown = autoshutdown
        self.meta = builder.meta if builder is not None else None
        self.sim_error = None       # (exception, traceback text) if the simulation thread failed
        self.finished = False       # the simulation ran to completion
        self.aborted = False        # the user quit before it finished
        self._latest = None
        self._applied = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._cycling = cycle is not None
        self._active = []           # views that apply to this run, in key order
        self._current = 0
        self._cycle_t = time.monotonic()

    # -- construction ------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield TextPane(id="topbar")
        yield Static(id="views-placeholder")  # replaced by the switcher once the meta is known
        yield TextPane(id="statusbar")
        yield Footer()

    def on_mount(self):
        self.set_interval(0.1, self._poll)
        # A daemon thread rather than a Textual worker: quitting must never wait on the simulation
        self._thread = threading.Thread(target=self._sim_worker, daemon=True, name="raps-sim")
        self._thread.start()
        if self.meta is not None:
            self._build_views(self.meta)

    def _build_views(self, meta):
        self._active = [v for v in VIEWS if v.applies(meta)]
        switcher = ContentSwitcher(*[v(id=f"view-{v.view_key}") for v in self._active], id="views",
                                   initial=f"view-{self._active[0].view_key}")
        self.query_one("#views-placeholder").remove()
        self.mount(switcher, before=self.query_one("#statusbar"))
        self._current = 0
        self._render_topbar()

    # -- simulation thread -------------------------------------------------------------------
    def _sim_worker(self):
        interval = 1.0 / self.hz
        last_tick = None
        try:
            next_due = 0.0
            stream = self.source if self.source is not None else self.engine.run_simulation(
                autoshutdown=self.autoshutdown)
            for item in stream:
                if self._stop.is_set():
                    break
                last_tick = item
                now = time.monotonic()
                if now >= next_due:
                    self._publish(item)
                    next_due = now + interval
            else:
                if last_tick is not None:
                    self._publish(last_tick)  # make sure the final state is shown
                self.finished = True
        except Exception as exc:
            self.sim_error = (exc, traceback.format_exc())

    def _publish(self, item):
        snap = self.builder.build(item) if self.builder is not None else item
        with self._lock:
            self._latest = snap

    # -- UI thread ---------------------------------------------------------------------------
    def _poll(self):
        with self._lock:
            snap = self._latest
        if snap is not None and snap is not self._applied:
            self._applied = snap
            self.apply_snapshot(snap)
        self._render_status()
        if self._cycling and self.cycle and len(self._active) > 1 and time.monotonic() - self._cycle_t >= self.cycle:
            self.action_next_view()

    def apply_snapshot(self, snap):
        if self.meta is None:
            self.meta = snap.meta
            self._build_views(snap.meta)
        view = self._current_view()
        if view is None:
            self._applied = None  # retry on the next poll once the views exist
            return
        view.update_snapshot(snap)
        if isinstance(self.screen, JobDetailScreen):
            self.screen.update_snapshot(snap)
        # Keep hidden views current so switching to one is instant and cheap
        for v in self.query("#views > View"):
            if v is not view:
                v.snapshot = snap

    def _current_view(self):
        if not self._active:
            return None
        try:
            return self.query_one(f"#view-{self._active[self._current].view_key}")
        except NoMatches:  # views are still being mounted
            return None

    def _render_topbar(self):
        t = Text()
        name = self.meta.system_name if self.meta else ""
        t.append(f" {name} ", style="bold")
        for i, v in enumerate(self._active):
            sel = i == self._current
            t.append(f" {v.view_key}:{v.title} ", style="reverse bold" if sel else "")
        if self._cycling:
            t.append(" [auto-cycle]", style="italic")
        self.query_one("#topbar", TextPane).update(t)

    def _banner(self):
        state = getattr(self.engine, "sim_state", None)
        if self.sim_error is not None:
            return (f"SIMULATION ERROR: {self.sim_error[0]!r}; press q to exit", "bold white on red")
        if self.finished:
            return ("Simulation complete: press q to exit", "bold black on green")
        b = state.banner() if state is not None else None
        return b

    def _render_status(self):
        snap = self._applied
        bar = Text()
        pane = self.query_one("#statusbar", TextPane)
        width = pane.content_size.width or max(self.size.width - 2, 1)  # minus CSS padding
        banner = self._banner()
        banner_text = f"  {banner[0]} " if banner else ""
        # The banner is never clipped: the progress bar shrinks (or disappears) to make room.
        # Measure in terminal cells, since the banner's emoji are double width.
        avail = width - cell_len(banner_text)
        if snap is None:
            bar.append(" waiting for the first snapshot...", style="dim")
        else:
            left = f" {snap.time_str}  {snap.rate_text}x  "
            right = f" {snap.progress * 100:5.1f}%"
            bw = avail - cell_len(left) - cell_len(right)
            if bw < 4:  # no room for a bar; keep the percentage if it fits
                bw = 0
                if cell_len(left) + cell_len(right) > avail:
                    right = ""
            if cell_len(left) <= avail:
                bar.append(left, style="bold")
            if bw:
                filled = int(bw * snap.progress)
                bar.append("█" * filled, style="green")
                bar.append("░" * (bw - filled), style="dim")
            bar.append(right)
        if banner:
            bar.append(banner_text, style=banner[1])
        pane.update(bar)

    # -- actions -----------------------------------------------------------------------------
    @property
    def sim_state(self):
        return getattr(self.engine, "sim_state", None)

    def action_pause(self):
        if self.sim_state is not None:
            self.sim_state.toggle_pause()

    def action_faster(self):
        if self.sim_state is not None:
            self.sim_state.speed_up()

    def action_slower(self):
        if self.sim_state is not None:
            self.sim_state.slow_down()

    def _show(self, index):
        if not self._active:
            return
        self._current = index % len(self._active)
        self._cycle_t = time.monotonic()
        view = self._active[self._current]
        self.query_one("#views", ContentSwitcher).current = f"view-{view.view_key}"
        self._render_topbar()
        cur = self._current_view()
        if cur is not None:
            if self._applied is not None:
                cur.update_snapshot(self._applied)
            cur.focus()

    def action_view_index(self, i: int):
        """Key `n` selects the view whose key is n, if that view applies to this run."""
        for pos, v in enumerate(self._active):
            if v is VIEWS[i]:
                self._show(pos)
                return

    def action_next_view(self):
        self._show(self._current + 1)

    def action_prev_view(self):
        self._show(self._current - 1)

    def action_toggle_cycle(self):
        self._cycling = not self._cycling
        if self._cycling and not self.cycle:
            self.cycle = 5.0
        self._cycle_t = time.monotonic()
        self._render_topbar()

    def open_job(self, job_id):
        """Show the detail modal for a job; the snapshot builder starts attaching its detail."""
        if self.builder is not None:
            self.builder.selected_job = job_id
        self.push_screen(JobDetailScreen(job_id))

    def show_job_in_map(self, job_id, rack):
        """Jump to the node map with the job's nodes highlighted and the cursor on one of its racks."""
        for pos, v in enumerate(self._active):
            if v.title == "Node Map":
                self._show(pos)
                self._current_view().show_job(job_id, rack)
                return

    def action_help(self):
        self.push_screen(HelpScreen())

    async def action_quit(self):
        if not self.finished and self.sim_error is None:
            self.aborted = True
        self._stop.set()
        if self.sim_state is not None and self.sim_state.is_paused():
            self.sim_state.toggle_pause()  # a paused engine never yields, so it could not see the stop flag
        self.exit()
