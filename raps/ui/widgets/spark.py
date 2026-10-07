"""A one-line labelled sparkline: name, trend, latest value."""
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, Sparkline

from raps.ui.widgets.textpane import TextPane


class LabeledSpark(Horizontal):
    DEFAULT_CSS = """
    LabeledSpark { height: 1; }
    LabeledSpark .spark-name { width: 12; color: $text-muted; }
    LabeledSpark Sparkline { width: 1fr; }
    LabeledSpark .spark-val { width: 14; height: 1; text-align: right; }
    """

    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self._label = name

    def compose(self) -> ComposeResult:
        yield Label(self._label, classes="spark-name")
        yield Sparkline([0.0], classes="spark-line")
        yield TextPane("", classes="spark-val")

    def set_data(self, values, text=""):
        spark = self.query_one(Sparkline)
        # Sparkline needs at least one point; keep the tail so it fills the available width
        spark.data = list(values[-400:]) or [0.0]
        self.query_one(".spark-val", TextPane).update(text)
