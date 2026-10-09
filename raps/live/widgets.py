"""Widgets and formatting for the live console: gradient bars, stat tiles and the node map."""
import colorsys
import hashlib
from datetime import datetime
from functools import lru_cache

import numpy as np
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Digits, Label

from raps.live.slurm import node_index
from raps.ui.widgets.pixelgrid import PixelGrid
from raps.ui.widgets.textpane import TextPane

# ---------------------------------------------------------------------------------------------
# palette (kept in sync with the theme in app.py)
# ---------------------------------------------------------------------------------------------
CYAN, VIOLET, PINK = "#38bdf8", "#a78bfa", "#f472b6"
GREEN, AMBER, RED, BLUE, SLATE = "#34d399", "#fbbf24", "#f87171", "#60a5fa", "#64748b"

STATE_STYLE = {
    "RUNNING": f"bold {GREEN}", "PENDING": f"bold {AMBER}", "COMPLETING": f"bold {CYAN}",
    "COMPLETED": GREEN, "FAILED": f"bold {RED}", "CANCELLED": SLATE, "TIMEOUT": f"bold {PINK}",
    "OUT_OF_MEMORY": f"bold {RED}", "NODE_FAIL": f"bold {RED}", "PREEMPTED": VIOLET,
    "REQUEUED": VIOLET, "REQUEUE_HOLD": VIOLET, "SUSPENDED": VIOLET,
}
STATE_ICON = {"RUNNING": "▶", "PENDING": "◷", "COMPLETING": "◌", "COMPLETED": "✔", "FAILED": "✖",
              "CANCELLED": "⊘", "TIMEOUT": "⌛", "OUT_OF_MEMORY": "✖", "NODE_FAIL": "✖"}


def state_text(state):
    return Text(f"{STATE_ICON.get(state, '•')} {state.lower()}", style=STATE_STYLE.get(state, ""))


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def lerp_hex(a, b, t):
    ra, rb = _hex(a), _hex(b)
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ra, rb))


def gradient(text, start=CYAN, end=VIOLET, bold=True):
    t = Text()
    n = max(len(text) - 1, 1)
    for i, ch in enumerate(text):
        t.append(ch, style=f"{'bold ' if bold else ''}{lerp_hex(start, end, i / n)}")
    return t


EIGHTHS = " ▏▎▍▌▋▊▉█"


def bar(frac, width, start=CYAN, end=VIOLET, empty="#1e293b"):
    """Smooth horizontal bar (1/8-cell precision) with a color gradient across the filled part."""
    frac = 0.0 if frac is None else min(max(frac, 0.0), 1.0)
    t = Text()
    if width <= 0:
        return t
    filled = frac * width
    full = int(filled)
    for i in range(full):
        t.append("█", style=lerp_hex(start, end, i / max(width - 1, 1)))
    if full < width:
        part = EIGHTHS[int((filled - full) * 8)]
        t.append(part, style=f"{lerp_hex(start, end, full / max(width - 1, 1))} on {empty}")
        t.append(" " * (width - full - 1), style=f"on {empty}")
    return t


def heat(frac):
    """Green -> amber -> red as a fraction grows."""
    frac = min(max(frac, 0.0), 1.0)
    return lerp_hex(GREEN, AMBER, frac * 2) if frac < 0.5 else lerp_hex(AMBER, RED, frac * 2 - 1)


def human_dur(seconds):
    if seconds is None:
        return "—"
    seconds = int(seconds)
    if seconds < 0:
        return "-" + human_dur(-seconds)
    d, r = divmod(seconds, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d{h:02d}h"
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def human_num(x, digits=1):
    ax = abs(x)
    if ax >= 1e6:
        return f"{x / 1e6:.{digits}f}M"
    if ax >= 1e4:
        return f"{x / 1e3:.{digits}f}k"
    return f"{x:,.0f}"


def rel_time(dt, now=None):
    """'in 2h05m' / '3m10s ago' / '—'."""
    if dt is None:
        return "—"
    delta = (dt - (now or datetime.now())).total_seconds()
    return f"in {human_dur(delta)}" if delta >= 0 else f"{human_dur(-delta)} ago"


@lru_cache(maxsize=8192)
def hashed_color(key, sat=0.65, light=0.58):
    """A stable, fairly bright color for an arbitrary string (job id, user, account)."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:6], 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, light, sat)
    return int(r * 255), int(g * 255), int(b * 255)


def spark(values, width, color=CYAN, lo=None, hi=None):
    """Unicode sparkline of the last `width` values, scaled to [lo, hi] (default: their own range)."""
    blocks = "▁▂▃▄▅▆▇█"
    vals = list(values)[-width:]
    t = Text()
    if not vals:
        return t.append("·" * width, style="#334155")
    lo = min(vals) if lo is None else lo
    hi = max(vals) if hi is None else hi
    span = hi - lo or 1
    t.append(" " * (width - len(vals)))
    for i, v in enumerate(vals):
        level = min(max(int((v - lo) / span * 7), 0), 7)
        t.append(blocks[level], style=lerp_hex("#1e3a5f", color, 0.35 + 0.65 * level / 7))
    return t


# ---------------------------------------------------------------------------------------------
# stat tile
# ---------------------------------------------------------------------------------------------


class Tile(Vertical):
    """A titled box with a big seven-segment number and a one-line caption under it."""

    DEFAULT_CSS = """
    Tile { width: 1fr; height: 7; border: round $primary-darken-1; padding: 0 1; background: $panel; }
    Tile > .tile-label { color: $text-muted; text-style: bold; width: 1fr; }
    Tile > Digits { width: auto; color: $primary; }
    Tile > .tile-sub { width: 1fr; height: 1; }
    Tile.warn > Digits { color: $warning; }
    Tile.bad > Digits { color: $error; }
    Tile.good > Digits { color: $success; }
    Tile.accent > Digits { color: $accent; }
    """

    def __init__(self, label, **kwargs):
        super().__init__(**kwargs)
        self._label = label

    def compose(self) -> ComposeResult:
        yield Label(self._label.upper(), classes="tile-label")
        yield Digits("—")
        yield TextPane("", classes="tile-sub")

    def set(self, value, sub=""):
        """``sub`` is a caption, or a list of captions from longest to shortest: the longest that fits is shown."""
        d = self.query_one(Digits)
        if d.value != str(value):
            d.update(str(value))
        self._subs = sub if isinstance(sub, (list, tuple)) else [sub]
        self._fit()

    def _fit(self):
        subs = getattr(self, "_subs", None)
        if not subs:
            return
        width = self.content_size.width
        cell = lambda s: s.cell_len if isinstance(s, Text) else len(s)  # noqa: E731
        self.query_one(".tile-sub", TextPane).update(next((s for s in subs if cell(s) <= width), subs[-1]))

    def on_resize(self, event):
        self._fit()


# ---------------------------------------------------------------------------------------------
# node map
# ---------------------------------------------------------------------------------------------
NODES_PER_CAB = 128

NODE_STATE_RGB = {
    "allocated": (52, 211, 153), "mixed": (52, 211, 153), "completing": (56, 189, 248),
    "idle": (30, 41, 59), "planned": (96, 165, 250), "reserved": (167, 139, 250),
    "maint": (251, 191, 36), "reboot": (251, 146, 60), "down": (248, 113, 113),
    "drained": (239, 68, 68), "draining": (244, 114, 182), "drain": (239, 68, 68),
    "fail": (220, 38, 38), "failing": (220, 38, 38), "future": (51, 65, 85), "unknown": (71, 85, 105),
}
STATE_GROUPS = [  # legend order: (label, states)
    ("busy", ("allocated", "mixed")), ("completing", ("completing",)), ("idle", ("idle",)),
    ("planned", ("planned",)), ("reserved", ("reserved",)), ("maint", ("maint", "reboot")),
    ("down", ("down", "drained", "draining", "drain", "fail", "failing")),
]
MINE_RGB = (244, 114, 182)
BG_RGB = (11, 15, 23)


def base_state(state):
    """'drained*' -> 'drained', 'allocated+' -> 'allocated'."""
    return state.rstrip("*$~#!%@^-+").lower()


def plan_cabinets(ncab, wpx, hpx):
    """
    Choose how to draw `ncab` cabinets of 128 nodes in wpx x hpx pixels with 1px gaps between
    cabinets. Returns (tw, th, k, cols, scale): a tile is tw x th node slots, each slot is
    scale x scale pixels and holds k nodes. Prefers big nodes (scale), then fewer nodes per pixel,
    then tiles that are taller than wide (so cabinets read as racks), then filling the area.
    None if nothing fits.
    """
    best = None
    for scale, k in [(s, 1) for s in (6, 5, 4, 3, 2)] + [(1, k) for k in (1, 2, 4, 8, 16, 32, 64, 128)]:
        per = NODES_PER_CAB // k
        for tw in (1, 2, 4, 8, 16, 32, 64, 128):
            if per % tw:
                continue
            th = per // tw
            pw, ph = tw * scale, th * scale
            cols = min((wpx + 1) // (pw + 1), ncab)
            if cols < 1:
                continue
            rows = -(-ncab // cols)
            if rows * (ph + 1) - 1 > hpx:
                continue
            fill = (cols * (pw + 1)) * (rows * (ph + 1)) / (wpx * hpx)
            score = (th >= tw, fill)
            if best is None or score > best[0]:
                best = (score, (tw, th, k, cols, scale))
        if best is not None:
            return best[1]
    return None


class NodeMap(PixelGrid):
    """
    Every node drawn as a pixel inside its cabinet (128 consecutive node numbers per cabinet).
    Hovering reports the node under the mouse; clicking posts ``NodeMap.Clicked``. Without a mouse
    (e.g. tmux with mouse off), the arrow keys move a highlighted node and enter opens its job.
    """

    BINDINGS = [
        Binding("left", "move(-1, 0)", "Node", show=False), Binding("right", "move(1, 0)", "Node", show=False),
        Binding("up", "move(0, -1)", "Node", show=False), Binding("down", "move(0, 1)", "Node", show=False),
        Binding("shift+left,pageup", "move(-128, 0)", "Cabinet", show=False),
        Binding("shift+right,pagedown", "move(128, 0)", "Cabinet", show=False),
        Binding("enter", "open", "Open job", show=False),
        Binding("escape", "clear_cursor", "Clear", show=False),
    ]

    can_focus = True  # so the Nodes tab's keys (c: color mode) reach it
    MODES = ("state", "job", "account", "mine", "power")
    # colors at fractions of the node maximum (RAPS power model)
    POWER_STOPS = ((0.0, (30, 41, 59)), (0.18, (14, 116, 144)), (0.43, (34, 197, 94)), (0.72, (245, 158, 11)),
                   (1.0, (239, 68, 68)))

    class Hover(Message):
        def __init__(self, nodes):
            super().__init__()
            self.nodes = nodes  # list of node names under the pointer ([] when off the map)

    class Clicked(Message):
        def __init__(self, nodes):
            super().__init__()
            self.nodes = nodes

    def __init__(self, **kwargs):
        super().__init__(source=self._draw, **kwargs)
        self.mode = "state"
        self.user = ""
        self.node_states = {}
        self.node_job = {}       # node -> Job
        self.node_power = {}     # node -> current watts
        self.max_w = 2800.0      # node maximum, scales the power colors
        self._names = []         # sorted node names
        self._cabs = []          # cabinet numbers present, in draw order
        self._pix_nodes = None   # (H, W) object array of node-name lists, for hover lookup
        self._pos = {}           # node -> (x, y, size) of its pixel block
        self._layout = None
        self.cursor = None       # node highlighted with the keyboard

    def set_data(self, node_states, node_job, user, mode=None, node_power=None, max_w=None):
        self.node_states = node_states or {}
        self.node_job = node_job or {}
        self.node_power = node_power or {}
        if max_w:
            self.max_w = max_w
        self.user = user
        if mode:
            self.mode = mode
        names = sorted(self.node_states, key=lambda n: (node_index(n) or 0, n))
        if names != self._names:
            self._names = names
            self._cabs = sorted({(node_index(n) or 1) - 1 >> 7 for n in names})
        self.refresh_image()

    def _node_rgb(self, name):
        st = base_state(self.node_states.get(name, "unknown"))
        job = self.node_job.get(name)
        rgb = NODE_STATE_RGB.get(st, NODE_STATE_RGB["unknown"])
        busy = st in ("allocated", "mixed", "completing")
        if self.mode == "power":
            w = self.node_power.get(name)
            return self.power_rgb(w / self.max_w) if w is not None else (51, 65, 85)
        if self.mode == "state":
            if busy and job is not None and job.user == self.user:
                return MINE_RGB
            return rgb
        if self.mode in ("job", "account"):
            if busy and job is not None:
                return hashed_color(job.id if self.mode == "job" else job.account)
            return rgb if not busy else (71, 85, 105)
        # mine: everything dim except my nodes
        if busy and job is not None and job.user == self.user:
            return MINE_RGB
        r, g, b = rgb
        return (r // 4 + 8, g // 4 + 10, b // 4 + 14)

    @classmethod
    def power_rgb(cls, frac):
        """Color for a node drawing `frac` of its maximum power."""
        stops = cls.POWER_STOPS
        if frac >= stops[-1][0]:
            return stops[-1][1]
        for (w0, c0), (w1, c1) in zip(stops, stops[1:]):
            if frac < w1:
                f = (frac - w0) / (w1 - w0)
                return tuple(round(a + (b - a) * f) for a, b in zip(c0, c1))
        return stops[0][1]

    def _draw(self, wpx, hpx):
        img = np.empty((hpx, wpx, 3), dtype=np.uint8)
        img[:] = BG_RGB
        self._pix_nodes = None
        if not self._names:
            return img
        plan = plan_cabinets(len(self._cabs), wpx, hpx)
        if plan is None:
            return img
        tw, th, k, cols, sc = plan
        self._layout = plan
        pw, ph = tw * sc, th * sc
        cab_pos = {c: i for i, c in enumerate(self._cabs)}
        acc = np.zeros((hpx, wpx, 3), dtype=np.float64)
        cnt = np.zeros((hpx, wpx), dtype=np.int32)
        pix_nodes = {}
        for name in self._names:
            idx = (node_index(name) or 1) - 1
            ci = cab_pos[idx >> 7]
            slot = (idx & 127) // k
            cx, cy = ci % cols, ci // cols
            x = cx * (pw + 1) + (slot % tw) * sc
            y = cy * (ph + 1) + (slot // tw) * sc
            if x + sc > wpx or y + sc > hpx:
                continue
            rgb = self._node_rgb(name)
            self._pos[name] = (x, y, sc)
            acc[y:y + sc, x:x + sc] += rgb
            cnt[y:y + sc, x:x + sc] += 1
            for yy in range(y, y + sc):
                for xx in range(x, x + sc):
                    pix_nodes.setdefault((yy, xx), []).append(name)
        has = cnt > 0
        img[has] = (acc[has] / cnt[has, None]).astype(np.uint8)
        # faint cabinet frames make the map read as racks even when mostly idle
        for ci in range(len(self._cabs)):
            cx, cy = ci % cols, ci // cols
            x0, y0 = cx * (pw + 1), cy * (ph + 1)
            empty = ~has[y0:y0 + ph, x0:x0 + pw]
            img[y0:y0 + ph, x0:x0 + pw][empty] = (20, 26, 38)
        self._pix_nodes = pix_nodes
        if self.cursor in self._pos and self.has_focus:
            x, y, s = self._pos[self.cursor]
            img[y:y + s, x:x + s] = (255, 255, 255)
        return img

    def action_move(self, dnode, drow):
        """Arrow keys: dnode steps along node numbers, drow steps a row of slots in the cabinet."""
        if not self._names:
            return
        if self.cursor not in self.node_states:
            self.cursor = self._names[0]
        else:
            k = self._layout[2] if self._layout else 1
            tw = self._layout[0] if self._layout else 1
            target = (node_index(self.cursor) or 0) + dnode * (k if abs(dnode) == 1 else 1) + drow * tw * k
            self.cursor = min(self._names, key=lambda n: abs((node_index(n) or 0) - target))
        self.refresh_image()
        self.post_message(self.Hover([self.cursor]))

    def action_clear_cursor(self):
        if self.cursor is not None:
            self.cursor = None
            self.refresh_image()
            self.post_message(self.Hover([]))

    def action_open(self):
        if self.cursor:
            self.post_message(self.Clicked([self.cursor]))

    def on_focus(self, event):
        self.refresh_image()

    def on_blur(self, event):
        self.refresh_image()

    def nodes_at(self, x, y):
        if not self._pix_nodes:
            return []
        return self._pix_nodes.get((2 * y, x)) or self._pix_nodes.get((2 * y + 1, x)) or []

    def on_mouse_move(self, event):
        self.post_message(self.Hover(self.nodes_at(event.x, event.y)))

    def on_leave(self, event):
        self.post_message(self.Hover([]))

    def on_click(self, event):
        nodes = self.nodes_at(event.x, event.y)
        if nodes:
            self.post_message(self.Clicked(nodes))

    def legend(self):
        t = Text()
        if self.mode == "power":
            for frac, _ in self.POWER_STOPS:
                w = frac * self.max_w
                t.append("■ ", style="#%02x%02x%02x" % self.power_rgb(frac))
                t.append(f"{w / 1000:.2f} kW max  " if frac == 1 else f"{w / 1000:.1f} kW  ", style="#94a3b8")
            vals = list(self.node_power.values())
            if vals:
                mean = sum(vals) / len(vals)
                t.append(f"total {sum(vals) / 1e6:.2f} MW · mean {mean:,.0f} W ({mean / self.max_w * 100:.0f}% of max)",
                         style="bold")
            else:
                t.append("no power readings yet", style="#64748b")
        elif self.mode in ("state", "mine"):
            counts = {}
            for st in self.node_states.values():
                counts[base_state(st)] = counts.get(base_state(st), 0) + 1
            mine = sum(1 for n, j in self.node_job.items() if j.user == self.user)
            groups = STATE_GROUPS if self.mode == "state" else []
            for label, states in groups:
                n = sum(counts.get(s, 0) for s in states)
                if n:
                    t.append("■ ", style="#%02x%02x%02x" % NODE_STATE_RGB[states[0]])
                    t.append(f"{label} ", style="#94a3b8")
                    t.append(f"{n:,} ", style="bold")
            t.append("■ ", style="#%02x%02x%02x" % MINE_RGB)
            t.append("mine ", style="#94a3b8")
            t.append(f"{mine:,}", style="bold")
        else:
            t.append(f"each color is one {self.mode}; ", style="#cbd5e1")
            t.append("■ ", style="#%02x%02x%02x" % NODE_STATE_RGB["idle"])
            t.append("idle  ", style="#cbd5e1")
            t.append("■ ", style="#%02x%02x%02x" % NODE_STATE_RGB["down"])
            t.append("down", style="#cbd5e1")
        if self._layout and self._layout[2] > 1:
            t.append(f"  {self._layout[2]}/px", style="dim")
        return t
