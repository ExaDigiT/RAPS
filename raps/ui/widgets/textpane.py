"""TextPane: a Static-like widget whose updates repaint without relaunching the layout."""
from rich.text import Text
from textual.widget import Widget


class TextPane(Widget):
    """
    Shows a rich ``Text`` (or markup string). Unlike ``Static.update``, which re-lays out the whole
    screen on every call, ``update`` only repaints unless the number of lines changed. The UI
    updates several of these several times a second, and layout passes were the dominant cost
    of the console when running next to the simulation thread.
    """

    DEFAULT_CSS = "TextPane { height: auto; }"

    def __init__(self, text="", **kwargs):
        super().__init__(**kwargs)
        self._text = Text()
        self._lines = 1
        self.update(text)

    def update(self, text=""):
        if isinstance(text, str):
            text = Text.from_markup(text)
        lines = max(text.plain.count("\n") + 1, 1)
        if text == self._text and lines == self._lines:
            return
        relayout = lines != self._lines
        self._text, self._lines = text, lines
        self.refresh(layout=relayout)

    def render(self):
        return self._text

    def get_content_height(self, container, viewport, width):
        return self._lines
