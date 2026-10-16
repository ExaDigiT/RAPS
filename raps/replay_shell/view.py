"""The shell as a tab of the simulation console (``RapsApp``), shown with ``raps run --shell``."""
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Input, Label, RichLog

from raps.ui.views.base import View


class ShellView(View):
    view_key = "7"
    title = "Shell"
    DEFAULT_CSS = """
    ShellView #rs-log { height: 1fr; border: round $primary-darken-2; padding: 0 1; }
    ShellView #rs-bar { dock: bottom; height: 3; border: tall $accent; }
    ShellView #rs-prompt { width: auto; padding: 0 1 0 1; color: $accent; text-style: bold; }
    ShellView #rs-input { border: none; height: 1; padding: 0; width: 1fr; }
    """

    def __init__(self, shell, **kwargs):
        super().__init__(**kwargs)
        self.shell = shell
        self.history = []
        self.hpos = None

    def compose(self) -> ComposeResult:
        log = RichLog(id="rs-log", wrap=True, markup=False, highlight=False, max_lines=5000)
        yield log
        with Horizontal(id="rs-bar"):
            yield Label(self.shell.prompt, id="rs-prompt")
            yield Input(placeholder="(try: help)", id="rs-input")

    def on_mount(self):
        self.query_one("#rs-log", RichLog).write(Text(
            f"~ is the sandbox: {self.shell.sandbox.root}\n"
            "Put job scripts there from outside (or write them here with echo > file).\n"
            "sbatch/squeue/sinfo act on the simulation, not the real machine. Type help.", style="dim"))
        self._prompt_title()

    def _prompt_title(self):
        self.query_one("#rs-log", RichLog).border_title = "simulated shell (nothing here touches the real machine)"
        self.query_one("#rs-prompt", Label).update(self.shell.prompt)

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
        self._prompt_title()
        if result.exit:
            self.app.action_view_index(0)   # leave the tab; the other views keep running
