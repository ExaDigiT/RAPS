"""The live console's Shell tab, driven by key presses in headless Textual against a fake Slurm."""
import asyncio
import time

import pytest

from raps.live.slurm import Slurm

from .test_live import FakeSlurm

pytestmark = pytest.mark.nodata
pytest.importorskip("textual")


def shell_app(tmp_path, monkeypatch):
    """A LiveApp whose shell starts in tmp_path, with a private HOME (.bashrc with an alias) and history."""
    from raps.live.app import LiveApp
    from raps.live.poller import Poller, Trend
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / ".bashrc").write_text("alias greet='echo hello-alias'\n")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.chdir(work)
    return LiveApp(Poller(FakeSlurm(), trend=Trend(None))), work


def log_text(app):
    from raps.live.views import Shell
    return "\n".join(line.text for line in app.query_one(Shell).query_one("#sh-log").lines)


async def wait_for(pilot, cond, seconds=10):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        await pilot.pause(0.1)
    return False


async def enter_shell(pilot):
    await pilot.pause(0.3)
    await pilot.press("6")
    await pilot.pause(0.3)


async def run_command(pilot, app, cmd):
    from textual.widgets import Input
    inp = app.query_one("#sh-input", Input)
    inp.value = cmd
    inp.cursor_position = len(cmd)
    await pilot.press("enter")


def test_commands_aliases_and_directory(tmp_path, monkeypatch):
    app, work = shell_app(tmp_path, monkeypatch)
    (work / "sub").mkdir()

    async def run():
        from raps.live.views import Shell
        async with app.run_test(size=(160, 48)) as pilot:
            await enter_shell(pilot)
            shell = app.query_one(Shell)
            await run_command(pilot, app, "echo plain-output")
            assert await wait_for(pilot, lambda: "exit 0" in log_text(app))
            assert "plain-output" in log_text(app)
            # aliases from ~/.bashrc work
            await run_command(pilot, app, "greet")
            assert await wait_for(pilot, lambda: "hello-alias" in log_text(app))
            # ls is laid out in columns on the pty, not one name per line
            for n in ("aa", "bb", "cc"):
                (work / n).write_text("")
            await run_command(pilot, app, "ls")
            assert await wait_for(pilot, lambda: "aa" in log_text(app).split("❯ ls")[-1])
            assert "aa" in log_text(app).split("❯ ls")[-1].split("\n")[1]
            assert "bb" in log_text(app).split("❯ ls")[-1].split("\n")[1]
            # the working directory and exports carry over to the next command
            await run_command(pilot, app, "cd sub && export FOO=bar")
            assert await wait_for(pilot, lambda: shell.cwd == work / "sub")
            await run_command(pilot, app, "pwd; echo foo=$FOO")
            assert await wait_for(pilot, lambda: "foo=bar" in log_text(app))
            assert str(work / "sub") in log_text(app)
            # plain cd (handled without bash) and cd -
            await run_command(pilot, app, "cd ..")
            assert await wait_for(pilot, lambda: shell.cwd == work)
            await run_command(pilot, app, "cd -")
            assert await wait_for(pilot, lambda: shell.cwd == work / "sub")

    asyncio.run(run())


def test_digits_switch_tab_only_on_an_empty_prompt(tmp_path, monkeypatch):
    app, _ = shell_app(tmp_path, monkeypatch)

    async def run():
        from textual.widgets import Input, TabbedContent
        async with app.run_test(size=(160, 48)) as pilot:
            await enter_shell(pilot)
            tabs = app.query_one(TabbedContent)
            inp = app.query_one("#sh-input", Input)
            assert tabs.active == "shell" and inp.has_focus
            for key, tab in (("3", "queue"), ("1", "overview"), ("5", "system"), ("7", "launch"), ("2", "myjobs")):
                if tabs.active != "shell":
                    await pilot.press("6")  # from another tab: back to the shell
                    await pilot.pause(0.2)
                assert tabs.active == "shell" and inp.value == ""
                await pilot.press(key)
                await pilot.pause(0.2)
                assert tabs.active == tab, (key, tabs.active)
            # 6 is the shell itself: typed into the prompt, not a switch
            await pilot.press("6")
            await pilot.pause(0.2)
            await pilot.press("6")
            assert tabs.active == "shell" and inp.value == "6"
            inp.value = ""
            # a digit after other text is just text
            await pilot.press("a", "1")
            assert tabs.active == "shell" and inp.value == "a1"
            # a lone digit entered with Enter switches too
            inp.value = "4"
            await pilot.press("enter")
            await pilot.pause(0.2)
            assert tabs.active == "nodes"

    asyncio.run(run())


def test_ctrl_k_kills_the_whole_command(tmp_path, monkeypatch):
    app, _ = shell_app(tmp_path, monkeypatch)

    async def run():
        from raps.live.views import Shell
        async with app.run_test(size=(160, 48)) as pilot:
            await enter_shell(pilot)
            shell = app.query_one(Shell)
            await pilot.press("ctrl+k")
            assert "nothing is running" in log_text(app)
            # a child of bash, too: the whole process group must go
            await run_command(pilot, app, "sleep 60 & sleep 61; echo not-reached")
            assert await wait_for(pilot, lambda: shell.proc is not None)
            pgid = shell.proc.pid
            await pilot.pause(0.5)
            t0 = time.monotonic()
            await pilot.press("ctrl+k")
            assert await wait_for(pilot, lambda: shell.proc is None, seconds=5)
            assert time.monotonic() - t0 < 5
            # "not-reached" is in the echoed command line once, and must not be in any output
            assert "killed" in log_text(app) and log_text(app).count("not-reached") == 1
            assert await wait_for(pilot, lambda: "exit -9" in log_text(app))
            import os
            with pytest.raises(ProcessLookupError):
                os.killpg(pgid, 0)
            # and the prompt is usable again
            await run_command(pilot, app, "echo after-kill")
            assert await wait_for(pilot, lambda: "after-kill" in log_text(app) and shell.proc is None)
            # ctrl+c kills as well
            await run_command(pilot, app, "sleep 60")
            assert await wait_for(pilot, lambda: shell.proc is not None)
            await pilot.press("ctrl+c")
            assert await wait_for(pilot, lambda: shell.proc is None, seconds=5)

    asyncio.run(run())


def test_tab_completes_in_the_prompt(tmp_path, monkeypatch):
    app, work = shell_app(tmp_path, monkeypatch)
    (work / "alpha_file.txt").write_text("")
    (work / "alpine").mkdir()
    (work / "alpine" / "inner.txt").write_text("")

    async def run():
        from textual.widgets import Input, TabbedContent
        async with app.run_test(size=(160, 48)) as pilot:
            await enter_shell(pilot)
            inp = app.query_one("#sh-input", Input)
            # unique match: completed and followed by a space
            inp.value = "cat alpha"
            inp.cursor_position = len(inp.value)
            await pilot.press("tab")
            assert inp.value == "cat alpha_file.txt "
            assert app.query_one(TabbedContent).active == "shell"  # Tab did not switch tabs
            # two matches: the common prefix, and the choices are listed
            inp.value = "cat al"
            inp.cursor_position = len(inp.value)
            await pilot.press("tab")
            assert inp.value == "cat alp"
            assert "alpha_file.txt" in log_text(app) and "alpine/" in log_text(app)
            # a directory completes with a slash and can be descended
            inp.value = "ls alpin"
            inp.cursor_position = len(inp.value)
            await pilot.press("tab")
            assert inp.value == "ls alpine/"
            await pilot.press("tab")
            assert inp.value == "ls alpine/inner.txt "
            # the first word completes as a command
            inp.value = "ech"
            inp.cursor_position = len(inp.value)
            await pilot.press("tab")
            assert inp.value.startswith("echo")
            # outside the shell Tab still switches tabs
            await pilot.press("escape")
            await pilot.press("tab")
            await pilot.pause(0.2)
            assert app.query_one(TabbedContent).active == "launch"

    asyncio.run(run())


def test_editor_replaces_vi_and_closes_with_escape(tmp_path, monkeypatch):
    app, work = shell_app(tmp_path, monkeypatch)
    (work / "COPYRIGHT").write_text("original\n")

    async def run():
        from textual.widgets import TextArea
        from raps.live.screens import EditScreen
        async with app.run_test(size=(160, 48)) as pilot:
            await enter_shell(pilot)
            await run_command(pilot, app, "vi notes.txt")
            await pilot.pause(0.3)
            assert isinstance(app.screen, EditScreen) and not app.screen.read_only
            await pilot.press("h", "i")
            assert app.screen.query_one(TextArea).text == "hi"
            await pilot.press("ctrl+s")
            assert (work / "notes.txt").read_text() == "hi"
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert not isinstance(app.screen, EditScreen)  # closed, and the app is still running
            assert app.is_running
            # unsaved changes need a second escape
            await run_command(pilot, app, "edit COPYRIGHT")
            await pilot.pause(0.3)
            assert isinstance(app.screen, EditScreen)
            await pilot.press("x")
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert isinstance(app.screen, EditScreen)
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert not isinstance(app.screen, EditScreen)
            assert (work / "COPYRIGHT").read_text() == "original\n"
            # less / view: read-only, ctrl+s writes nothing
            await run_command(pilot, app, "less COPYRIGHT")
            await pilot.pause(0.3)
            assert isinstance(app.screen, EditScreen) and app.screen.read_only
            await pilot.press("x")
            await pilot.press("ctrl+s")
            await pilot.press("escape")
            await pilot.pause(0.2)
            assert (work / "COPYRIGHT").read_text() == "original\n"
            # programs that need a real terminal get a message, not a hang
            await run_command(pilot, app, "top")
            assert await wait_for(pilot, lambda: "needs a full terminal" in log_text(app))
            await run_command(pilot, app, "vi")
            assert await wait_for(pilot, lambda: "give one file name" in log_text(app))

    asyncio.run(run())


def test_status_line_and_relaxed_polling(tmp_path, monkeypatch):
    app, _ = shell_app(tmp_path, monkeypatch)

    async def run():
        async with app.run_test(size=(160, 48)) as pilot:
            await pilot.pause(0.5)
            assert app.poller.relax == 1.0
            await pilot.press("6")
            await pilot.pause(0.3)
            assert app.poller.relax == 4.0
            assert await wait_for(pilot, lambda: app.poller.latest().jobs is not None)
            await pilot.pause(1.5)  # the bars redraw once a second
            status = str(app.query_one("#statusbar").render())
            assert "running" in status and "pending" in status and "power" in status and "you:" in status
            assert "fetching" not in status and "●" not in status  # no per-source timers
            await pilot.press("3")
            await pilot.pause(1.5)
            assert app.poller.relax == 1.0
            assert "jobs" in str(app.query_one("#statusbar").render())  # the timers are back on other tabs

    asyncio.run(run())
