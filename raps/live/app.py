"""
LiveApp: a Textual console for the real Frontier — queue, my jobs, node map, system and a shell.

Run with ``python main.py live`` (or ``python -m raps.live``). Slurm is polled in a background
thread (see ``poller.py``); the UI only renders the latest published ``ClusterState``.
"""
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta

from rich.text import Text
from textual.app import App, ComposeResult
from textual.actions import SkipAction
from textual.binding import Binding
from textual.theme import Theme
from textual.widgets import Footer, TabbedContent, TabPane

from raps.live.fastforward import MAX_SPAN, DEFAULT_SPEED, SPEEDS, Playback, fmt_offset, fmt_speed
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
    #ffbar { height: 1; display: none; background: #2e1065; }
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
        Binding("x", "forecast_run", "24 h run", show=False),
        Binding("f", "ff_key", "Fast-forward (ff)", show=False),
        Binding("l,plus,equals_sign", "ff_speed(1)", "Faster", show=False),   # the keys of `raps run`
        Binding("j,minus,underscore", "ff_speed(-1)", "Slower", show=False),
        Binding("k,space", "ff_pause", "Pause", show=False),
        Binding("0", "ff_reset_speed", "Reset speed", show=False),
        Binding("less_than_sign", "ff_restart_key", "Restart (<<)", show=False),
        Binding("escape", "ff_stop", "Back to live", show=False),
    ] + [Binding(str(i + 1), f"tab({i})", title, show=False) for i, (_, title, _) in enumerate(VIEWS)] \
      + [Binding(f"f{i + 1}", f"tab({i})", title, show=False, priority=True) for i, (_, title, _) in enumerate(VIEWS)]

    def __init__(self, poller: Poller | None = None, start_tab="overview", **kwargs):
        super().__init__(**kwargs)
        self.poller = poller or Poller()
        self.start_tab = start_tab
        self._applied = None
        self._tick = 0
        self.ff = None            # Playback while fast-forwarding, else None
        self.ff_t = 0.0           # simulated seconds from now
        self.ff_speed = DEFAULT_SPEED
        self.ff_paused = False
        self._ff_lt = 0.0         # monotonic time of the last `<`
        self._ff_busy = False     # the simulation is running
        self._ff_last = 0.0       # monotonic time of the last `f` and of the last ff frame
        self._ff_frame = 0.0

    def compose(self) -> ComposeResult:
        yield TextPane("", id="topbar")
        yield TextPane("", id="ffbar")
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
        self.set_interval(0.5, self._ff_step)

    def on_unmount(self):
        self.poller.stop()

    # -- state flow ---------------------------------------------------------------------------
    def _poll(self):
        if self.ff is not None:
            return
        state = self.poller.latest()
        if state is not self._applied:
            self._applied = state
            for _, _, cls in VIEWS:
                for v in self.query(cls):
                    v.apply(state, self.poller.trend)
            self._render_bars()

    def on_tabbed_content_tab_activated(self, event):
        self.poller.relax_polling(event.pane.id == "shell")
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
        left = gradient(f"◆ {(state.cluster if state and state.cluster else self.poller.system or 'frontier').upper()}", CYAN, VIOLET)
        if self.ff is not None:
            left.append("  ")
            left.append(" ⏩ FORECAST ", style="bold #ffffff on #7c3aed")
        else:
            left.append("  ")
            left.append(" ● LIVE ", style=f"bold #0b0f17 on {GREEN}")
        left.append("  ·  exadigit", style="#64748b")
        right = Text()
        if state is not None:
            right.append(f"{state.user}", style=f"bold {PINK}")
            right.append(f"@{state.cluster or 'frontier'}", style="#94a3b8")
        shown = datetime.now() + timedelta(seconds=self.ff_t) if self.ff is not None else datetime.now()
        right.append(f"   {shown:%a %b %d  %H:%M:%S}", style="bold #ddd6fe" if self.ff is not None else "bold #e2e8f0")
        pad = max(width - left.cell_len - right.cell_len, 1)
        topbar = self.query_one("#topbar", TextPane)
        topbar.styles.background = "#1e1b4b" if self.ff is not None else "#0f1a2b"
        topbar.update(left + Text(" " * pad) + right)
        self._render_ffbar(width)

        if self.query_one(TabbedContent).active == "shell":
            self.query_one("#statusbar", TextPane).update(self._shell_stats(state))
            return
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

    def _render_ffbar(self, width):
        """The banner under the top bar while fast-forwarding: where we are on the forecast timeline."""
        bar = self.query_one("#ffbar", TextPane)
        bar.display = self.ff is not None
        if self.ff is None:
            return
        end = max(self.ff.end, 1)
        when = datetime.now() + timedelta(seconds=self.ff_t)
        head = Text()
        head.append(" ⏩ THE FUTURE, NOT LIVE ", style="bold #ffffff on #7c3aed")
        head.append(f" {when:%a %b %d %H:%M} ", style="bold #ddd6fe")
        head.append(f" {fmt_offset(self.ff_t)} ", style="bold #c4b5fd")
        tail = Text()
        tail.append(f"  {'⏸ paused' if self.ff_paused else '▶ ' + fmt_speed(SPEEDS[self.ff_speed])}", style="bold #ddd6fe")
        tail.append(f"   of {fmt_offset(end)}   j/l speed · k pause · << restart · esc live ", style="#a5b4fc")
        room = max(width - head.cell_len - tail.cell_len - 2, 10)
        filled = min(int(room * self.ff_t / end), room)
        track = Text("█" * filled, style="#a78bfa") + Text("░" * (room - filled), style="#4c1d95")
        bar.update(head + Text(" ") + track + tail)

    def _shell_stats(self, state):
        """Status line for the Shell tab: what the machine is doing, instead of when each source was fetched."""
        from raps.live.analysis import summarize
        t = Text()
        if state is None:
            return Text("waiting for Slurm …", style="#64748b")
        if state.jobs is not None:
            sm = summarize(state)
            t.append(f"{sm.run_jobs:,} running", style=f"bold {GREEN}")
            t.append(f" ({sm.run_nodes:,} nodes)  ", style="#94a3b8")
            t.append(f"{sm.pend_jobs:,} pending", style=f"bold {AMBER}")
            t.append(f" ({sm.eligible_jobs:,} eligible)  ", style="#94a3b8")
            if sm.total_nodes:
                t.append(f"nodes {sm.busy_nodes:,}/{sm.usable_nodes:,} busy ({sm.utilization:.0%})", style="#e2e8f0")
                t.append(f" · {sm.idle_nodes:,} idle", style="#94a3b8")
                if sm.out_nodes:
                    t.append(f" · {sm.out_nodes:,} out", style=RED)
                t.append("  ")
            if state.power is not None:
                t.append(f"power {state.power / 1e6:.2f} MW", style=f"bold {PINK}")
                if state.machine_max_w:
                    t.append(f" ({state.power / state.machine_max_w:.0%} of max)", style="#94a3b8")
                t.append("  ")
            t.append(f"you: {sm.my_run} running · {sm.my_pend} pending", style=PINK)
        else:
            t.append("waiting for squeue …", style="#64748b")
        for name, err in state.errors.items():
            t.append(f"   ✖ {name}: {err[:40]}", style=f"bold {RED}")
        t.append(f"   (polling {self.poller.relax:g}× slower in the shell)", style="#475569")
        return t

    # -- actions ------------------------------------------------------------------------------
    def action_tab(self, i):
        self.query_one(TabbedContent).active = VIEWS[i][0]

    def _step_tab(self, d):
        keys = [k for k, _, _ in VIEWS]
        tc = self.query_one(TabbedContent)
        tc.active = keys[(keys.index(tc.active) + d) % len(keys)]

    def action_next_tab(self):
        focused = self.focused
        if focused is not None and focused.id == "sh-input":  # in the shell, Tab completes (F-keys still switch tabs)
            self.query_one("#view-shell").complete(focused)
            return
        self._step_tab(1)

    def action_prev_tab(self):
        self._step_tab(-1)

    def open_shell_with(self, cmd, note="add your batch script (or --wrap \"...\") and press enter"):
        """Switch to the Shell tab with `cmd` typed at the prompt, ready to edit and run."""
        from textual.widgets import Input
        self.query_one(TabbedContent).active = "shell"
        inp = self.query_one("#sh-input", Input)
        inp.value = cmd
        inp.cursor_position = len(cmd)
        inp.focus()
        self.notify(note, timeout=8)

    def action_forecast_run(self):
        """`x`: write the queue as a RAPS replay (jobs.npz + experiment.yaml) and put `raps run` at the Shell prompt."""
        state = self.poller.latest() if self.ff is not None else self._applied   # the real queue, not the future
        if state is None or state.jobs is None:
            self.notify("waiting for the queue", severity="warning")
            return
        self.notify(f"exporting {len(state.jobs):,} jobs as a 24 h RAPS run …")
        threading.Thread(target=self._export_forecast_run, args=(state, self.poller.system or "frontier"),
                         daemon=True).start()

    # -- fast-forward (`ff`) --------------------------------------------------------------------
    def ff_key(self):
        """An `f` outside the Queue's filter key; True when it is the second of `ff`."""
        now = time.monotonic()
        double = now - self._ff_last < 0.6
        self._ff_last = 0.0 if double else now
        if double:
            self.start_ff()
        return double

    def action_ff_key(self):
        focused = self.focused
        if focused is not None and focused.id == "sh-input":
            raise SkipAction()
        if not self.ff_key() and self.ff is None:
            self.notify("press f again to fast-forward", timeout=1.5)

    def start_ff(self):
        if self.ff is not None:
            self.action_ff_stop()
            return
        state = self.poller.latest()
        if state is None or state.jobs is None or state.node_states is None:
            self.notify("waiting for the queue and nodes", severity="warning")
            return
        if self._ff_busy:
            return
        self._ff_busy = True
        self.notify(f"simulating {len(state.jobs):,} jobs into the future …")
        threading.Thread(target=self._ff_simulate, args=(state,), daemon=True).start()

    def _ff_simulate(self, state):
        from raps.live.twin import simulate
        try:
            pn = (self.poller.twin.partition_nodes if self.poller.twin is not None else None) \
                or self.poller.slurm.partition_nodes()
            res = simulate(state, pn, horizon=MAX_SPAN, placement="random")
        except Exception as exc:
            self._ff_busy = False
            self.call_from_thread(self.notify, f"fast-forward failed: {exc}", severity="error", timeout=10)
            return
        self.call_from_thread(self._ff_begin, Playback(state, res))

    def _ff_begin(self, playback):
        self._ff_busy = False
        self.ff, self.ff_t, self.ff_paused = playback, 0.0, False
        self._ff_frame = time.monotonic()
        self.notify("fast-forward: j/l (or - / +) speed, k (or space) pause, 0 reset speed, esc back to live, ff again also leaves", timeout=6)
        self._ff_show()

    def _ff_show(self):
        self._applied = self.ff.state_at(self.ff_t)
        for _, _, cls in VIEWS:
            for v in self.query(cls):
                v.apply(self._applied, self.poller.trend)
        self._render_bars()

    def _ff_step(self):
        if self.ff is None:
            return
        now = time.monotonic()
        dt, self._ff_frame = now - self._ff_frame, now
        if self.ff_paused:
            return
        self.ff_t = min(self.ff_t + dt * SPEEDS[self.ff_speed], self.ff.end)
        if self.ff_t >= self.ff.end:
            self.ff_paused = True
            self.notify("fast-forward: the queue has drained", timeout=5)
        self._ff_show()

    def action_ff_speed(self, d):
        if self.ff is not None:
            self.ff_speed = min(max(self.ff_speed + d, 0), len(SPEEDS) - 1)
            self._render_bars()

    def action_ff_restart_key(self):
        """`<<`: back to the start of the forecast and play it again."""
        if self.ff is None:
            raise SkipAction()
        now = time.monotonic()
        if now - self._ff_lt < 0.6:
            self._ff_lt = 0.0
            self.ff_t, self.ff_paused, self._ff_frame = 0.0, False, now
            self._ff_show()
            self.notify("fast-forward restarted", timeout=1.5)
        else:
            self._ff_lt = now
            self.notify("press < again to restart from now", timeout=1.5)

    def action_ff_reset_speed(self):
        if self.ff is not None:
            self.ff_speed = DEFAULT_SPEED
            self._render_bars()

    def action_ff_pause(self):
        if self.ff is not None:
            self.ff_paused = not self.ff_paused
            if self.ff_t >= self.ff.end and not self.ff_paused:
                self.ff_t = 0.0
            self._render_bars()

    def action_ff_stop(self):
        if self.ff is None:
            raise SkipAction()
        self.ff = None
        self._applied = None   # _poll re-applies the live state
        self.notify("back to live", timeout=2)
        self._poll()

    def _export_forecast_run(self, state, system):
        from raps.live.forecast_run import export
        try:
            folder, yaml, rep = export(state, system)
        except Exception as exc:
            self.call_from_thread(self.notify, f"export failed: {exc}", severity="error", timeout=10)
            return
        lost = f", {rep['dropped']} did not fit" if rep["dropped"] else ""
        self.call_from_thread(
            self.open_shell_with, f"raps run {yaml} --noui",
            f"{rep['running']} running + {rep['queued']} queued jobs{lost} in {folder} "
            f"(also {folder.parent / 'latest'}); enter runs it, edit the yaml first to change policy, time or cooling")

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


CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def nodes_per_rack(system, default=128):
    """nodes_per_rack from config/<system>.yaml (read directly: loading the full config takes ~5 s)."""
    try:
        import yaml
        with open(CONFIG_DIR / f"{system}.yaml") as f:
            return int(yaml.safe_load(f)["system"]["nodes_per_rack"])
    except Exception:
        return default


def main(argv=None):
    import argparse
    from raps.live import live_add_parser
    parser = argparse.ArgumentParser(prog="raps")
    live_add_parser(parser.add_subparsers(required=True))
    args = parser.parse_args(["live", *(argv if argv is not None else __import__("sys").argv[1:])])
    run(args)


def run(args):
    import threading
    from raps.live import say
    from raps.live.slurm import Slurm
    cluster = []
    if args.system is None:  # ask Slurm which machine this is while the rest loads (scontrol can take seconds)
        threading.Thread(target=lambda: cluster.append(Slurm.cluster_name()), daemon=True).start()
    from raps.live.forecast import Forecaster
    from raps.live.widgets import set_nodes_per_cab
    system = args.system
    if system is None:  # the real machine's own config, e.g. lux -> config/lux.yaml
        say("asking Slurm which cluster this is")
        while not cluster:
            threading.Event().wait(0.05)
        system = cluster[0] if cluster[0] and (CONFIG_DIR / f"{cluster[0]}.yaml").exists() else "frontier"
    set_nodes_per_cab(nodes_per_rack(system))
    say("opening the console")
    forecaster = Forecaster(days=args.forecast_days) if args.forecast_days > 0 else None
    poller = Poller(Slurm(user=args.user, history_days=args.history_days), scale=args.slow, forecaster=forecaster,
                    twin=not args.no_twin, system=system or None, access=True)
    say("")
    LiveApp(poller, start_tab=args.tab).run()

