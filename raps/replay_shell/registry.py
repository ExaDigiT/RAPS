"""
Command registry and the shell session that dispatches to it.

A command is a function ``handler(ctx, args) -> int`` (exit status) registered with a name, a one-line
summary and a usage string. It writes with ``ctx.out`` / ``ctx.err`` and reaches the filesystem only
through ``ctx.sandbox`` and the simulation only through ``ctx.adapter``. There is no subprocess
execution anywhere: a name that is not registered is "command not found".

``ReplayShell.run(line)`` parses with ``shlex``, handles ``>``/``>>`` redirection of a command's output
into a sandbox file, runs the handler and returns a ``Result``. It is UI-free so tests (and a plain
terminal REPL) can drive it; ``view.ShellView`` is the Textual front end.

TODO(intern): pipes (``|``), ``;`` and ``&&``, ``$VAR`` expansion, glob expansion (``*.sh``), history,
tab completion (``CommandRegistry.names`` and ``Sandbox.resolve`` give the candidates).
"""
import shlex
from dataclasses import dataclass, field
from typing import Callable

from raps.replay_shell.sandbox import Sandbox, SandboxError


@dataclass
class Command:
    name: str
    handler: Callable
    summary: str
    usage: str = ""


class CommandRegistry:
    def __init__(self):
        self._cmds = {}

    def register(self, name, summary, usage=""):
        """Decorator: ``@registry.register("ls", "list files", "ls [-l] [path ...]")``."""
        def deco(fn):
            self._cmds[name] = Command(name, fn, summary, usage or name)
            return fn
        return deco

    def add(self, command):
        self._cmds[command.name] = command

    def get(self, name):
        return self._cmds.get(name)

    @property
    def names(self):
        return sorted(self._cmds)

    def __iter__(self):
        return iter(sorted(self._cmds.values(), key=lambda c: c.name))


class ExitShell(Exception):
    """Raised by ``exit`` to end the session."""


class QuitConsole(ExitShell):
    """Raised by ``quit``: end the session and ask the host (the RAPS console) to quit too."""


@dataclass
class Result:
    output: str = ""
    status: int = 0
    exit: bool = False
    quit: bool = False   # also quit the console


@dataclass
class Context:
    sandbox: Sandbox
    adapter: object            # SimAdapter or None when no simulation is attached
    registry: CommandRegistry
    _out: list = field(default_factory=list)
    _err: list = field(default_factory=list)

    def out(self, text=""):
        self._out.append(str(text))

    def err(self, text):
        self._err.append(str(text))

    def need_sim(self):
        """The adapter, or a clear error for commands that cannot work without a simulation."""
        if self.adapter is None:
            raise CommandError("no simulation attached")
        return self.adapter


class CommandError(Exception):
    """A handler's way to fail with a message: the session prints it and returns status 1."""


class ReplayShell:
    def __init__(self, sandbox, adapter, registry):
        self.sandbox = sandbox
        self.adapter = adapter
        self.registry = registry

    @property
    def prompt(self):
        return f"{self.sandbox.pwd} $ "

    def run(self, line):
        line = line.strip()
        if not line or line.startswith("#"):
            return Result()
        try:
            words = shlex.split(line)
        except ValueError as exc:
            return Result(f"shell: {exc}\n", 2)
        redirect = None
        for i, w in enumerate(words):
            if w in (">", ">>"):
                if i != len(words) - 2:
                    return Result("shell: syntax error near redirection\n", 2)
                redirect, words = (w, words[i + 1]), words[:i]
                break
        if not words:
            return Result()
        cmd = self.registry.get(words[0])
        if cmd is None:
            return Result(f"{words[0]}: command not found (try help)\n", 127)
        ctx = Context(self.sandbox, self.adapter, self.registry)
        status, leave, quit_ = 0, False, False
        try:
            status = cmd.handler(ctx, words[1:]) or 0
        except ExitShell as exc:
            leave, quit_ = True, isinstance(exc, QuitConsole)
        except (CommandError, SandboxError) as exc:
            ctx.err(f"{words[0]}: {exc}")
            status = 1
        except NotImplementedError as exc:
            ctx.err(f"{words[0]}: not implemented yet" + (f" ({exc})" if str(exc) else ""))
            status = 1
        out = "".join(s if s.endswith("\n") else s + "\n" for s in ctx._out)
        if redirect and not leave:
            try:
                path = self.sandbox.resolve(redirect[1])
                if path.is_dir():
                    raise CommandError(f"{redirect[1]}: is a directory")
                with open(path, "a" if redirect[0] == ">>" else "w") as f:
                    f.write(out)
                out = ""
            except (CommandError, SandboxError, OSError) as exc:
                ctx.err(f"shell: {exc}")
                status = 1
        err = "".join(s if s.endswith("\n") else s + "\n" for s in ctx._err)
        return Result(out + err, status, leave, quit_)
