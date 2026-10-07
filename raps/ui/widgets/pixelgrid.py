"""PixelGrid: draws an RGB image with half-block characters, two pixels per terminal cell."""
import numpy as np
from rich.segment import Segment
from rich.style import Style
from textual.color import Color
from textual.strip import Strip
from textual.widget import Widget

_STYLE_CACHE: dict[int, Style] = {}


def _style(pair):
    """Style for a packed (top rgb, bottom rgb) pair: top is the glyph color, bottom the background."""
    st = _STYLE_CACHE.get(pair)
    if st is None:
        if len(_STYLE_CACHE) > 50000:
            _STYLE_CACHE.clear()
        top, bot = pair >> 24, pair & 0xFFFFFF
        st = Style(color=Color(top >> 16, (top >> 8) & 255, top & 255).rich_color,
                   bgcolor=Color(bot >> 16, (bot >> 8) & 255, bot & 255).rich_color)
        _STYLE_CACHE[pair] = st
    return st


def pack_rgb(img):
    """uint8 [..., 3] to int64 [...] as 0xRRGGBB."""
    a = img.astype(np.int64)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


class PixelGrid(Widget):
    """
    Renders an image of ``width x (2 * height)`` pixels. The owner supplies a ``source`` callable
    ``(width_px, height_px) -> uint8[height_px, width_px, 3]`` that is invoked on every
    ``refresh_image()`` and whenever the widget is resized, so the picture always fits the
    terminal regardless of the data's own shape.
    """

    DEFAULT_CSS = """
    PixelGrid { width: 1fr; height: 1fr; min-height: 3; }
    """

    def __init__(self, source=None, **kwargs):
        super().__init__(**kwargs)
        self.source = source
        self._img = None
        self._packed = None

    @property
    def pixel_size(self):
        return self.size.width, self.size.height * 2

    def set_source(self, source):
        self.source = source
        self.refresh_image()

    def set_image(self, img):
        self._img = img
        self._packed = pack_rgb(img) if img is not None else None
        self.refresh()

    def refresh_image(self):
        w, h = self.pixel_size
        if self.source is None or w <= 0 or h <= 0:
            return
        self.set_image(self.source(w, h))

    def on_resize(self, event):
        self.refresh_image()

    def render_line(self, y):
        width = self.size.width
        p = self._packed
        if p is None or p.shape[1] != width or 2 * y >= p.shape[0]:
            return Strip.blank(width)
        top = p[2 * y]
        bot = p[2 * y + 1] if 2 * y + 1 < p.shape[0] else np.zeros_like(top)
        pair = (top << 24) | bot
        edges = np.flatnonzero(pair[1:] != pair[:-1]) + 1
        starts = np.concatenate(([0], edges))
        ends = np.concatenate((edges, [width]))
        segs = [Segment("▀" * int(e - s), _style(int(pair[s]))) for s, e in zip(starts, ends)]
        return Strip(segs, width)
