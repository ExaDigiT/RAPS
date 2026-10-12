"""Modal screens: help, confirmation, the full job detail card, and a small text editor."""
import re
from pathlib import Path

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from raps.live.widgets import CYAN, VIOLET, gradient, state_text

HELP = """\
[b]Views[/b]
  1-7 / F1-F7   Overview · My Jobs · Queue · Nodes · System · Shell · Launch
  tab / S-tab   next / previous view
  r             refresh everything now
  ?             this help        q   quit

[b]My Jobs[/b]   enter details   c cancel   h hold   u release   (all ask first)
[b]Queue[/b]     / filter   esc clear   f running/pending/all   m only mine   s sort   enter details
[b]Maps[/b]      c color mode (state, job, account, mine, power) on Overview and Nodes
          hover a node, or on Nodes use the arrow keys: [ ] (or shift+←/→, PgUp/PgDn)
          jump a cabinet, shift+↑/↓ a row of cabinets, home/end the first/last node;
          click or enter opens the node's job. In tmux, hover and click need
          `set -g mouse on`.   hover a node · click for its job
[b]24 h run[/b] x anywhere: write the queue as a RAPS replay (jobs.npz + experiment.yaml, in
          ~/raps-forecast-runs, newest as `latest`) and put `raps run <yaml> --noui` at the Shell prompt
[b]Fast-forward[/b] ff (f twice) plays the twin's forecast of the queue on the existing tabs, jobs starting
          and ending, nodes and power changing; j/l slower/faster, k pause, 0 reset speed, << restart from now (as in raps run; - + and space work too),
          esc (or ff) back to live.
          No new submissions. x still exports the real queue.
[b]Shell[/b]     type any command; ↑/↓ history; ctrl+k kills it; `clear` clears;
          esc leaves the prompt (then digits switch tabs; F1-F7 always do)
[b]Launch[/b]    where and when could my job run? Type nodes and time, a: next project,
          t / enter: ask Slurm (sbatch --test-only, nothing is submitted) and the twin;
          enter on a row puts its sbatch line in the Shell tab. "you ✓/✗" is what Slurm
          says for each project (some partitions, e.g. g1, are for member projects only).

[b]Forecast[/b]  runtime and W/node from the last 5 finished runs of the same job (●, same user
          and job name), else the same user (◐), else the same project (○). Pending jobs show
          the forecast runtime, running jobs the forecast time left. Built from 30 days of
          sacct -a, kept in memory only (--forecast-days, 0 turns it off).

[b]Power[/b]     system power = the sum of Slurm's CurrentWatts over all compute nodes (no
          cooling, network or storage), every minute; readings above 5 kW are glitches and are
          dropped. "W now" is a running job's measured W/node, "W/node" its forecast.
          The twin's power forecast keeps running jobs at their measured power, gives
          starting jobs their forecast (else the running median) and free nodes the idle mean.
          Colors and "% of max" use the node maximum from RAPS's power model
          (compute_node_power at 100% CPU/GPU/NIC: 2,773 W on Frontier; --system).

[b]Twin start[/b]  every 2 min the queue is run forward through RAPS's scheduler (priority
          order with backfill, planning with time limits as Slurm does) while jobs end at
          their forecast runtimes. "slurm" is Slurm's own estimate, which assumes every job
          runs to its limit. Not modelled: new submissions, priority aging, dependencies,
          per-user limits, future reservations. --no-twin turns it off.

Data refreshes in the background: squeue every 15 s, node states every 60 s,
history and the twin every 2 min, allocations, fair share and forecasts every 10 min.
"""


class HelpScreen(ModalScreen):
    BINDINGS = [Binding("escape,question_mark,q", "app.pop_screen", "Close")]
    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > Vertical { width: 92; max-width: 100%; height: auto; max-height: 100%;
                            border: thick $primary; background: $surface; padding: 1 2; }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(gradient("◆ LIVE · keys", CYAN, VIOLET))
            yield Static(HELP)


class ConfirmScreen(ModalScreen[bool]):
    BINDINGS = [Binding("escape,n", "no", "No"), Binding("y", "yes", "Yes")]
    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    ConfirmScreen > Vertical { width: 70; max-width: 100%; height: auto; border: thick $error;
                               background: $surface; padding: 1 2; }
    ConfirmScreen Horizontal { height: auto; margin-top: 1; align-horizontal: right; }
    ConfirmScreen Button { margin-left: 2; }
    """

    def __init__(self, message, **kwargs):
        super().__init__(**kwargs)
        self.message = message

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(self.message)
            with Horizontal():
                yield Button("No (n)", id="no")
                yield Button("Yes (y)", id="yes", variant="error")

    def on_button_pressed(self, event):
        self.dismiss(event.button.id == "yes")

    def action_yes(self):
        self.dismiss(True)

    def action_no(self):
        self.dismiss(False)


def parse_scontrol(text):
    """'Key=Value Key2=Value with spaces' records -> ordered dict (values may contain spaces)."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # keys are CamelCase words followed by '='; values run until the next key
        for m in re.finditer(r"(\w[\w:/]*)=(.*?)(?=\s+\w[\w:/]*=|$)", line):
            out[m.group(1)] = m.group(2)
    return out


DETAIL_GROUPS = [
    ("Job", ["JobId", "JobName", "UserId", "Account", "Partition", "QOS", "JobState", "Reason",
             "Priority", "Dependency"]),
    ("Size", ["NumNodes", "NumCPUs", "NumTasks", "ReqTRES", "AllocTRES", "Features", "Reservation"]),
    ("Time", ["SubmitTime", "EligibleTime", "StartTime", "EndTime", "RunTime", "TimeLimit"]),
    ("Where", ["NodeList", "BatchHost", "WorkDir", "Command", "StdOut", "StdErr"]),
]


class EditScreen(ModalScreen):
    """The shell's ``edit``/``vi FILE`` editor: ctrl+s saves, esc closes (twice to drop changes); read-only for less."""

    BINDINGS = [Binding("ctrl+s", "save", "Save", priority=True), Binding("escape", "close", "Close", priority=True),
                Binding("f10", "close", "Close", show=False, priority=True)]
    DEFAULT_CSS = """
    EditScreen { align: center middle; }
    EditScreen > Vertical { width: 100%; height: 100%; background: $surface; }
    EditScreen #ed-bar { height: 1; padding: 0 1; background: $panel; }
    EditScreen TextArea { height: 1fr; }
    """

    def __init__(self, path, read_only=False, **kwargs):
        super().__init__(**kwargs)
        self.path = Path(path)
        self.read_only = read_only
        self.saved = ""
        self.closing = False
        self.note = ""

    def compose(self) -> ComposeResult:
        try:
            self.saved = self.path.read_text() if self.path.exists() else ""
        except (OSError, UnicodeDecodeError) as exc:
            self.saved = ""
            self.note = f"could not read: {exc}"
        with Vertical():
            yield Static("", id="ed-bar")
            yield TextArea(self.saved, show_line_numbers=True, tab_behavior="indent", read_only=self.read_only, id="ed-text")

    def on_mount(self):
        self.query_one(TextArea).focus()
        self._bar(self.note)

    def on_text_area_changed(self, event):
        self.closing = False
        self._bar()

    def _bar(self, note=""):
        dirty = self.query_one(TextArea).text != self.saved
        t = Text(f"{'view' if self.read_only else 'edit'} {self.path}", style="bold")
        t.append("  [modified]" if dirty else "", style="bold #fbbf24")
        keys = "esc close" if self.read_only else "ctrl+s save · esc close"
        t.append(f"   {keys}{' · ' + note if note else ''}", style="#64748b")
        self.query_one("#ed-bar", Static).update(t)

    def action_save(self):
        if self.read_only:
            return
        text = self.query_one(TextArea).text
        try:
            self.path.write_text(text)
        except OSError as exc:
            return self._bar(f"not saved: {exc}")
        self.saved = text
        self.closing = False
        self._bar("saved")

    def action_close(self):
        if self.query_one(TextArea).text != self.saved and not self.closing:
            self.closing = True
            return self._bar("unsaved changes: esc again to discard")
        self.dismiss()


class JobDetailScreen(ModalScreen):
    BINDINGS = [Binding("escape,q,enter", "app.pop_screen", "Close")]
    DEFAULT_CSS = """
    JobDetailScreen { align: center middle; }
    JobDetailScreen > VerticalScroll { width: 110; max-width: 100%; height: auto; max-height: 90%;
                                       border: thick $primary; background: $surface; padding: 1 2; }
    """

    def __init__(self, job_id, fetch, **kwargs):
        super().__init__(**kwargs)
        self.job_id = job_id
        self.fetch = fetch

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(gradient(f"◆ job {self.job_id}", CYAN, VIOLET))
            yield Static(Text("loading scontrol show job …", style="dim"), id="detail-body")

    def on_mount(self):
        self.load()

    @work(thread=True, exclusive=True)
    def load(self):
        try:
            body = render_detail(parse_scontrol(self.fetch(self.job_id)))
        except Exception as exc:
            body = Text(f"scontrol failed: {exc}", style="bold red")
        self.app.call_from_thread(self.query_one("#detail-body", Static).update, body)


def render_detail(kv):
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold #94a3b8", no_wrap=True)
    grid.add_column(overflow="fold")
    for title, keys in DETAIL_GROUPS:
        present = [k for k in keys if kv.get(k) not in (None, "", "(null)", "None")]
        if not present:
            continue
        grid.add_row(Text(title.upper(), style=f"bold {VIOLET}"), "")
        for k in present:
            v = kv[k]
            if len(v) > 600:
                v = v[:600] + " …"
            grid.add_row(k, state_text(v) if k == "JobState" else Text(v))
        grid.add_row("", "")
    return grid
