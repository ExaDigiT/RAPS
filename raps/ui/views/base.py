"""Base class and small formatting helpers shared by the views."""
from rich.text import Text
from textual.containers import Vertical

from raps.utils import convert_seconds_to_hhmmss


class View(Vertical):
    """
    One switchable screen of the console. Subclasses set ``view_key``/``title``, override
    ``applies`` to hide themselves when the run has no data for them, and implement
    ``update_snapshot``, which the app calls with the latest snapshot while the view is visible
    (and once more when it becomes visible).
    """

    view_key = "?"
    title = "View"
    DEFAULT_CSS = """
    View { width: 1fr; height: 1fr; }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.snapshot = None

    @classmethod
    def applies(cls, meta) -> bool:
        return True

    def update_snapshot(self, snap):
        self.snapshot = snap

    def on_show(self):
        if self.snapshot is not None:
            self.update_snapshot(self.snapshot)


def fmt_dur(seconds):
    return convert_seconds_to_hhmmss(int(seconds))


def fmt_num(x, digits=1):
    """Compact number: 12.3k, 4.5M."""
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.{digits}f}G"
    if ax >= 1e6:
        return f"{x / 1e6:.{digits}f}M"
    if ax >= 1e4:
        return f"{x / 1e3:.{digits}f}k"
    return f"{x:,.0f}" if ax >= 100 else f"{x:.{digits}f}"


def kv_lines(pairs, key_style="dim", val_style="bold"):
    """Rich Text of `key  value` lines for a tile."""
    t = Text()
    for i, (k, v) in enumerate(pairs):
        if i:
            t.append("\n")
        t.append(f"{k:<10}", style=key_style)
        t.append(str(v), style=val_style)
    return t


def sync_table(table, keys, rows):
    """
    Make a DataTable show `rows` (tuples of cells), one per unique string key in `keys`.

    If the keys match the table's current rows in order, only the changed cells are updated, which
    repaints without a layout pass; otherwise the table is rebuilt (cursor and scroll are the
    caller's concern). Returns True if it was rebuilt.
    """
    prev_keys = getattr(table, "_sync_keys", None)
    if prev_keys == keys and table.row_count == len(keys):
        col_keys = list(table.columns)
        for key, old, new in zip(keys, table._sync_rows, rows):
            for ck, o, n in zip(col_keys, old, new):
                if o != n:
                    table.update_cell(key, ck, n, update_width=False)
        table._sync_rows = rows
        return False
    table.clear()
    for key, row in zip(keys, rows):
        table.add_row(*row, key=key)
    table._sync_keys, table._sync_rows = list(keys), rows
    return True
