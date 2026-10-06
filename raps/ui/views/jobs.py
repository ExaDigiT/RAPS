"""Jobs: every running and queued job in a scrollable, sortable, filterable table."""
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Input
from raps.ui.widgets.textpane import TextPane

from raps.ui.views.base import View, fmt_dur, sync_table

# Sort keys: column title -> index into the snapshot job tuple
_COLUMNS = [("JOBID", 0), ("NAME", 1), ("ACCOUNT", 2), ("ST", 3), ("NODES", 4),
            ("TIME LIMIT", 5), ("WALL TIME", 6), ("SLOWDOWN", 7)]


class Jobs(View):
    view_key = "2"
    title = "Jobs"
    BINDINGS = [
        Binding("slash", "filter", "Filter", show=False),
        Binding("s", "sort", "Sort column", show=False),
        Binding("r", "reverse", "Reverse sort", show=False),
        Binding("escape", "clear_filter", "Clear filter", show=False),
    ]
    DEFAULT_CSS = """
    Jobs #jobs-filter { height: 3; display: none; }
    Jobs #jobs-filter.shown { display: block; }
    Jobs #jobs-info { height: 1; color: $text-muted; }
    Jobs #jobs-table { height: 1fr; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sort_col = None       # index into the visible columns, None keeps scheduler order
        self.sort_desc = False
        self.filter_text = ""
        self._cols = None

    def compose(self) -> ComposeResult:
        yield Input(placeholder="filter by id, name or account (Enter to apply, Esc to clear)", id="jobs-filter")
        yield TextPane(id="jobs-info")
        yield DataTable(id="jobs-table", cursor_type="row", zebra_stripes=True)

    def _setup(self, snap):
        net = snap.meta.has_network
        self._cols = [c for c in _COLUMNS if c[0] != "SLOWDOWN" or net]
        self.query_one("#jobs-table", DataTable).add_columns(*[c[0] for c in self._cols])

    def on_show(self):
        super().on_show()
        self.query_one("#jobs-table").focus()

    def action_filter(self):
        box = self.query_one("#jobs-filter", Input)
        box.add_class("shown")
        box.focus()

    def action_clear_filter(self):
        box = self.query_one("#jobs-filter", Input)
        box.value = ""
        box.remove_class("shown")
        self.filter_text = ""
        self.query_one("#jobs-table").focus()
        self._refresh()

    def on_input_changed(self, event: Input.Changed):
        self.filter_text = event.value.strip().lower()
        self._refresh()

    def on_input_submitted(self, event: Input.Submitted):
        self.query_one("#jobs-table").focus()

    def action_sort(self):
        n = len(self._cols or [])
        self.sort_col = 0 if self.sort_col is None else (self.sort_col + 1) % n
        self._refresh()

    def action_reverse(self):
        self.sort_desc = not self.sort_desc
        self._refresh()

    def update_snapshot(self, snap):
        super().update_snapshot(snap)
        if self._cols is None:
            self._setup(snap)
        self._refresh()

    def _refresh(self):
        snap = self.snapshot
        if snap is None or self._cols is None:
            return
        t = self.query_one("#jobs-table", DataTable)
        rows = snap.jobs
        if self.filter_text:
            f = self.filter_text
            rows = [r for r in rows if f in str(r[0]).lower() or f in r[1].lower() or f in r[2].lower()]
        if self.sort_col is not None:
            idx = self._cols[self.sort_col][1]
            rows = sorted(rows, key=lambda r: r[idx], reverse=self.sort_desc)
        # Keep the cursor on the same job across refreshes
        sel = None
        if t.row_count and t.cursor_row < t.row_count:
            try:
                sel = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
            except Exception:
                sel = None
        scroll = t.scroll_y
        net = snap.meta.has_network
        out = []
        for r in rows:
            cells = [str(r[0]).zfill(5), r[1], r[2], r[3], str(r[4]), fmt_dur(r[5]), fmt_dur(r[6])]
            if net:
                cells.append(f"{r[7]:.2f}x")
            # dilated by network congestion: yellow
            out.append(tuple(Text(c, style="yellow" if r[8] else "") for c in cells))
        rebuilt = sync_table(t, [str(r[0]) for r in rows], out)
        if rebuilt and sel is not None:
            try:
                t.move_cursor(row=t.get_row_index(sel), animate=False)
            except Exception:
                pass
        if rebuilt:
            t.scroll_y = scroll
        sort = ("none" if self.sort_col is None
                else f"{self._cols[self.sort_col][0]} {'desc' if self.sort_desc else 'asc'}")
        shown = f"{len(rows)} of {snap.n_running + snap.n_queued}"
        trunc = " (list capped)" if snap.jobs_truncated else ""
        self.query_one("#jobs-info", TextPane).update(
            f"{snap.n_running} running, {snap.n_queued} queued | showing {shown}{trunc} | sort: {sort} | "
            f"/ filter  s sort  r reverse")
