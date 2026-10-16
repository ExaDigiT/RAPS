"""The shell as a tab of the simulation console (``RapsApp``), shown with ``raps run --shell``."""
from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Input, RichLog

from raps.ui.views.base import View


class ShellView(View):
    view_key = "7"
    title = "Shell"
    DEFAULT_CSS = """
    ShellView #rs-log { height: 1fr; border: round $primary-darken-2; padding: 0 1; }
    ShellView #rs-input { dock: bottom; border: tall $accent; }
    """

    def __init__(self, shell, **kwargs):
        super().__init__(**kwargs)
        self.shell = shell
        self.history = []
        self.hpos = None

    def compose(self) -> ComposeResult:
        log = RichLog(id="rs-log", wrap=True, markup=False, highlight=False, max_lines=5000)
        log.border_title = "simulated shell (nothing here touches the real machine)"
        yield log
        yield Input(placeholder=self.shell.prompt + "(try: help)", id="rs-input")

    def on_mount(self):
        self.query_one("#rs-log", RichLog).write(Text(
            f"Sandbox: {self.shell.sandbox.root}\nsbatch/squeue/sinfo act on the simulation. Type help.", style="dim"))

    def on_show(self):
        super().on_show()
        self.query_one("#rs-input", Input).focus()

    def on_input_submitted(self, event):
        line = event.value
        event.input.value = ""
        log = self.query_one("#rs-log", RichLog)
        log.write(Text(self.shell.prompt + line, style="bold"))
        result = self.shell.run(line)
        if result.output:
            log.write(Text(result.output.rstrip("\n")))
        event.input.placeholder = self.shell.prompt
        if result.exit:
            self.app.action_view_index(0)   # leave the tab; the other views keep running
