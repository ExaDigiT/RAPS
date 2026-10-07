"""Paired bar chart of running vs queued jobs by size (number of nodes), in power-of-two buckets."""
import math

import numpy as np
from rich.text import Text

from raps.ui.widgets.textpane import TextPane

_EIGHTHS = " ▁▂▃▄▅▆▇█"
RUN_STYLE, QUEUE_STYLE = "#40c86e", "#e6b422"


def size_buckets(total_nodes):
    """Upper bounds 1, 2, 4, 8, ... covering total_nodes; bucket i holds sizes in (2^(i-1), 2^i]."""
    return [1 << i for i in range(max(math.ceil(math.log2(max(total_nodes, 2))), 1) + 1)]


def bucket_counts(sizes, n_buckets):
    sizes = np.asarray(sizes, dtype=np.int64)
    if sizes.size == 0:
        return np.zeros(n_buckets, dtype=np.int64)
    idx = np.ceil(np.log2(np.maximum(sizes, 1))).astype(np.int64).clip(0, n_buckets - 1)
    return np.bincount(idx, minlength=n_buckets)


def fmt_bound(n):
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def render_hist(run, queued, labels, width, height):
    """
    Text of `height` lines: height-1 rows of bars (running then queued side by side per bucket) and
    one row of bucket labels, laid out to fit `width` columns.
    """
    rows = max(height - 1, 1)
    n = len(labels)
    slot = max(width // n, 2)
    bar = max((slot - 1) // 2, 1)
    peak = max(int(run.max(initial=0)), int(queued.max(initial=0)), 1)

    def column(v):
        eighths = int(round(v / peak * rows * 8))
        if v > 0:
            eighths = max(eighths, 1)
        return eighths

    cols = [(column(int(r)), column(int(q))) for r, q in zip(run, queued)]
    out = Text()
    for row in range(rows - 1, -1, -1):
        for r8, q8 in cols:
            used = 0
            for e, style in ((r8, RUN_STYLE), (q8, QUEUE_STYLE)):
                ch = _EIGHTHS[min(max(e - row * 8, 0), 8)]
                out.append(ch * bar, style=style)
                used += bar
            out.append(" " * (slot - used))
        out.append("\n")
    for lab in labels:
        out.append(lab[:slot - 1].center(slot - 1) + " ", style="dim")
    return out


class SizeHistogram(TextPane):
    DEFAULT_CSS = "SizeHistogram { height: 1fr; }"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._data = None

    def set_data(self, run_sizes, queue_sizes, total_nodes):
        bounds = size_buckets(total_nodes)
        self._data = (bucket_counts(run_sizes, len(bounds)), bucket_counts(queue_sizes, len(bounds)),
                      [fmt_bound(b) for b in bounds])
        self._draw()

    def on_resize(self, event):
        self._draw()

    def _draw(self):
        w, h = self.size.width, self.size.height
        if self._data is None or w < 10 or h < 3:
            return
        run, queued, labels = self._data
        self.update(render_hist(run, queued, labels, w, h))
        self.set_subtitle(run, queued)

    def set_subtitle(self, run, queued):
        self.border_subtitle = f"max {max(int(run.max(initial=0)), int(queued.max(initial=0)))} jobs per bucket"
