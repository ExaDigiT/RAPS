"""Modal screens: help, confirmation, and the full job detail card."""
import re

from rich.table import Table
from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from raps.live.widgets import CYAN, VIOLET, gradient, state_text

HELP = """\
[b]Views[/b]
  1-6 / F1-F6   Overview · My Jobs · Queue · Nodes · System · Shell
  tab / S-tab   next / previous view
  r             refresh everything now
  ?             this help        q   quit

[b]My Jobs[/b]   enter details   c cancel   h hold   u release   (all ask first)
[b]Queue[/b]     / filter   esc clear   f running/pending/all   m only mine   s sort   enter details
[b]Nodes[/b]     c color mode (state, job, account, mine, power)   hover a node · click for its job
[b]Shell[/b]     type any command; ↑/↓ history; ctrl+k kills it; `clear` clears;
          esc leaves the prompt (then 1-5 switch tabs; F1-F6 always do)

[b]Forecast[/b]  runtime and W/node from the last 5 finished runs of the same job (●, same user
          and job name), else the same user (◐), else the same project (○). Pending jobs show
          the forecast runtime, running jobs the forecast time left. Built from 30 days of
          sacct -a, kept in memory only (--forecast-days, 0 turns it off).

[b]Power[/b]     measured: Slurm's CurrentWatts per node (compute nodes only; no cooling,
          network or storage), every minute; readings above 5 kW are glitches and are
          dropped. "W now" is a running job's measured W/node, "W/node" its forecast.
          The twin's power forecast keeps running jobs at their measured power, gives
          starting jobs their forecast (else the running median) and free nodes the idle mean.
          Nodes view: c cycles to the power color mode.

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
            yield Static(gradient("◆ FRONTIER LIVE · keys", CYAN, VIOLET))
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
