"""
LiveApp: a Textual console for the real Frontier — queue, my jobs, node map, system and a shell.

Run with ``python main.py live`` (or ``python -m raps.live``). Slurm is polled in a background
thread (see ``poller.py``); the UI only renders the latest published ``ClusterState``.
"""
import subprocess
from datetime import datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.theme import Theme
from textual.widgets import Footer, TabbedContent, TabPane

from raps.live.poller import Poller
from raps.live.screens import ConfirmScreen, HelpScreen, JobDetailScreen
from raps.live.widgets import AMBER, CYAN, GREEN, PINK, RED, VIOLET, gradient, human_num, rel_time
from raps.live.views import VIEWS
from raps.ui.widgets.textpane import TextPane

FRONTIER_THEME = Theme(
    name="frontier",
    primary=CYAN, secondary=VIOLET, accent=PINK, warning=AMBER, error=RED, success=GREEN,
    foreground="#e2e8f0", background="#0b0f17", surface="#111827", panel="#0f1623", dark=True,
    variables={"footer-key-foreground": CYAN, "block-cursor-background": "#1e3a5f",
               "block-cursor-foreground": "#f8fafc", "input-selection-background": "#38bdf855"},
)

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class LiveApp(App):
    TITLE = "Frontier Live"
    CSS = """
    Screen { background: $background; }
    #topbar { height: 1; background: #0f1a2b; padding: 0 1; }
    #statusbar { height: 1; padding: 0 1; background: #0f1a2b; }
    TabbedContent { height: 1fr; }
    TabbedContent ContentSwitcher { height: 1fr; }
    TabPane { padding: 0; }
    Tabs { background: #0b0f17; }
    DataTable > .datatable--header { background: #131c2c; color: $secondary; text-style: bold; }
    """
    BINDINGS = [
        Binding("r", "refresh", "Refresh"),
        Binding("question_mark", "help", "Help"),
        Binding("q", "quit", "Quit"),
        # priority: otherwise Tab moves focus between widgets once a view has something focusable
        Binding("tab", "next_tab", "Next", show=False, priority=True),
        Binding("shift+tab", "prev_tab", "Prev", show=False, priority=True),
        Binding("c", "color_mode", "Map colors", show=False),
    ] + [Binding(str(i + 1), f"tab({i})", title, show=False) for i, (_, title, _) in enumerate(VIEWS)] \
      + [Binding(f"f{i + 1}", f"tab({i})", title, show=False, priority=True) for i, (_, title, _) in enumerate(VIEWS)]

    def __init__(self, poller: Poller | None = None, start_tab="overview", **kwargs):
        super().__init__(**kwargs)
        self.poller = poller or Poller()
        self.start_tab = start_tab
        self._applied = None
        self._tick = 0

    def compose(self) -> ComposeResult:
        yield TextPane("", id="topbar")
        with TabbedContent(initial=self.start_tab, id="tabs"):
            for i, (key, title, cls) in enumerate(VIEWS):
                with TabPane(f"{i + 1} {title}", id=key):
                    yield cls(id=f"view-{key}")
        yield TextPane("", id="statusbar")
        yield Footer()

    def on_mount(self):
        self.register_theme(FRONTIER_THEME)
        self.theme = "frontier"
        self.poller.start()
        self.set_interval(0.25, self._poll)
        self.set_interval(1.0, self._render_bars)

    def on_unmount(self):
        self.poller.stop()

    # -- state flow ---------------------------------------------------------------------------
    def _poll(self):
        state = self.poller.latest()
        if state is not self._applied:
            self._applied = state
            for _, _, cls in VIEWS:
                for v in self.query(cls):
                    v.apply(state, self.poller.trend)
            self._render_bars()

    def on_tabbed_content_tab_activated(self, event):
        if self._applied is not None:
            view = event.pane.query_children().first()
            view.apply(self._applied, self.poller.trend)
        if event.pane.id != "shell":
            self.set_focus(None)
            tables = event.pane.query("DataTable")
            if tables:
                tables.first().focus()
            elif event.pane.query("NodeMap"):
                event.pane.query("NodeMap").first().focus()

    def _render_bars(self):
        self._tick += 1
        state = self._applied
        width = self.size.width - 2
        left = gradient("◆ FRONTIER", CYAN, VIOLET) + Text("  live", style=f"bold {PINK}")
        left.append("  ·  exadigit", style="#64748b")
        right = Text()
        if state is not None:
            right.append(f"{state.user}", style=f"bold {PINK}")
            right.append(f"@{state.cluster or 'frontier'}", style="#94a3b8")
        right.append(f"   {datetime.now():%a %b %d  %H:%M:%S}", style="bold #e2e8f0")
        pad = max(width - left.cell_len - right.cell_len, 1)
        self.query_one("#topbar", TextPane).update(left + Text(" " * pad) + right)

        t = Text()
        busy = sorted(self.poller.busy)
        if busy:
            t.append(f"{SPINNER[self._tick % len(SPINNER)]} fetching {', '.join(busy)}  ", style=CYAN)
        if state is not None:
            for name in ("jobs", "nodes", "power", "history", "allocations", "forecast", "twin"):
                upd = state.updated.get(name)
                if name in state.errors:
                    t.append(f"✖ {name}: {state.errors[name][:50]}  ", style=f"bold {RED}")
                elif upd:
                    t.append(f"● {name} ", style=GREEN)
                    t.append(f"{rel_time(upd)}  ", style="#64748b")
                else:
                    t.append(f"○ {name}  ", style="#475569")
        rule = state.forecast if state is not None else None
        if rule is not None:
            t.append(f" history rule: {human_num(rule.n_runs)} runs / {rule.days} d", style="#64748b")
        elif self.poller.forecaster is None:
            t.append(" forecasts off", style="#475569")
        tw = state.twin if state is not None else None
        if tw is not None:
            t.append(f"   RAPS twin: {tw.simulated:,} jobs in {tw.seconds:.1f} s", style="#64748b")
        elif self.poller.twin is None:
            t.append("   RAPS twin off", style="#475569")
        self.query_one("#statusbar", TextPane).update(t)

    # -- actions ------------------------------------------------------------------------------
    def action_tab(self, i):
        self.query_one(TabbedContent).active = VIEWS[i][0]

    def _step_tab(self, d):
        keys = [k for k, _, _ in VIEWS]
        tc = self.query_one(TabbedContent)
        tc.active = keys[(keys.index(tc.active) + d) % len(keys)]

    def action_next_tab(self):
        self._step_tab(1)

    def action_prev_tab(self):
        self._step_tab(-1)

    def open_shell_with(self, cmd):
        """Switch to the Shell tab with `cmd` typed at the prompt, ready to edit and run."""
        from textual.widgets import Input
        self.query_one(TabbedContent).active = "shell"
        inp = self.query_one("#sh-input", Input)
        inp.value = cmd
        inp.cursor_position = len(cmd)
        inp.focus()
        self.notify("add your batch script (or --wrap \"...\") and press enter", timeout=6)

    def action_color_mode(self):
        """`c` on a tab with a node map (Overview, Nodes), whatever has focus; My Jobs keeps c = cancel."""
        view = self.query_one(TabbedContent).active_pane.query_children().first()
        if hasattr(view, "action_color_mode"):
            view.action_color_mode()

    def action_refresh(self):
        self.poller.refresh()
        self.notify("refreshing all sources", timeout=2)

    def action_help(self):
        self.push_screen(HelpScreen())

    def show_job(self, job_id):
        self.push_screen(JobDetailScreen(job_id, self.poller.slurm.job_detail))

    def open_node_job(self, nodes):
        from raps.live.analysis import node_job_map
        jobs = node_job_map((self._applied.jobs or []) if self._applied else [])
        job = next((jobs[n] for n in nodes if n in jobs), None)
        if job is None:
            self.notify(f"{nodes[0]}: no running job", timeout=3)
        else:
            self.show_job(job.id)

    def confirm_and_run(self, message, cmd):
        def done(ok):
            if not ok:
                return
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0:
                self.notify(f"{' '.join(cmd)}: done", severity="information")
            else:
                self.notify((p.stderr or p.stdout).strip()[:200] or "failed", title=" ".join(cmd), severity="error")
            self.poller.refresh("jobs", "history")
        self.push_screen(ConfirmScreen(message), done)


def main(argv=None):
    import argparse
    from raps.live import live_add_parser
    parser = argparse.ArgumentParser(prog="raps")
    live_add_parser(parser.add_subparsers(required=True))
    args = parser.parse_args(["live", *(argv if argv is not None else __import__("sys").argv[1:])])
    run(args)


def run(args):
    from raps.live.forecast import Forecaster
    from raps.live.slurm import Slurm
    forecaster = Forecaster(days=args.forecast_days) if args.forecast_days > 0 else None
    poller = Poller(Slurm(user=args.user, history_days=args.history_days), scale=args.slow, forecaster=forecaster,
                    twin=not args.no_twin, system=args.system or None, access=True)
    LiveApp(poller, start_tab=args.tab).run()

