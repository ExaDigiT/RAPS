"""Filesystem and session built-ins, implemented in Python on top of ``Sandbox`` (no subprocesses).

TODO(intern): touch, rm (-r), cp, mv, head, tail, wc, grep, a tiny line editor (``edit job.sh``) or a
Textual TextArea screen for writing job scripts, and ``ls -l`` sizes/mtimes.
"""
from raps.replay_shell.registry import CommandError, ExitShell


def _pwd(ctx, args):
    ctx.out(ctx.sandbox.pwd)


def _cd(ctx, args):
    if len(args) > 1:
        raise CommandError("too many arguments")
    ctx.sandbox.chdir(args[0] if args else "~")   # bare `cd` goes to the sandbox root, our "home"


def _ls(ctx, args):
    long = "-l" in args
    paths = [a for a in args if not a.startswith("-")] or ["."]
    for i, p in enumerate(paths):
        real = ctx.sandbox.resolve(p)
        if not real.exists():
            raise CommandError(f"{p}: no such file or directory")
        if real.is_dir():
            if len(paths) > 1:
                ctx.out(("\n" if i else "") + f"{p}:")
            names = sorted(c.name + ("/" if c.is_dir() else "") for c in real.iterdir())
            if long:
                for n in names:
                    ctx.out(n)
            elif names:
                ctx.out("  ".join(names))
        else:
            ctx.out(real.name)


def _cat(ctx, args):
    if not args:
        raise CommandError("usage: cat file ...")
    for p in args:
        real = ctx.sandbox.resolve(p)
        if not real.is_file():
            raise CommandError(f"{p}: no such file")
        ctx.out(real.read_text(errors="replace").rstrip("\n"))


def _mkdir(ctx, args):
    parents = "-p" in args
    names = [a for a in args if a != "-p"]
    if not names:
        raise CommandError("usage: mkdir [-p] dir ...")
    for p in names:
        try:
            ctx.sandbox.resolve(p).mkdir(parents=parents, exist_ok=parents)
        except FileExistsError:
            raise CommandError(f"{p}: file exists")
        except FileNotFoundError:
            raise CommandError(f"{p}: no such directory (use -p)")


def _echo(ctx, args):
    ctx.out(" ".join(args))


def _help(ctx, args):
    if args:
        cmd = ctx.registry.get(args[0])
        if cmd is None:
            raise CommandError(f"no such command: {args[0]}")
        ctx.out(f"{cmd.usage}\n  {cmd.summary}")
        return
    width = max(len(c.name) for c in ctx.registry)
    for c in ctx.registry:
        ctx.out(f"  {c.name:<{width}}  {c.summary}")


def _exit(ctx, args):
    raise ExitShell


def register_all(r):
    r.register("pwd", "print the working directory (~ is the sandbox root)", "pwd")(_pwd)
    r.register("cd", "change directory inside the sandbox", "cd [dir]")(_cd)
    r.register("ls", "list files", "ls [-l] [path ...]")(_ls)
    r.register("cat", "print files", "cat file ...")(_cat)
    r.register("mkdir", "make directories", "mkdir [-p] dir ...")(_mkdir)
    r.register("echo", "print text (use > file or >> file to write a script)", "echo text ...")(_echo)
    r.register("help", "list commands, or describe one", "help [command]")(_help)
    r.register("exit", "leave the shell", "exit")(_exit)
