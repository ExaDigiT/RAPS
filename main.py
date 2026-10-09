#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
ExaDigiT Resource Allocator & Power Simulator (RAPS)
"""
import argparse
from pathlib import Path
import os
import sys
import textwrap
import copy
import gzip
import threading


class Splash:
    """
    Something pretty to look at while `raps live` loads (standard library only, as it starts before the slow
    imports): a braille spinner that cycles through the console's colors, the title shimmering beside it, and
    the current step. Drawn on stderr from a thread; ``stop`` clears the line and brings the cursor back.
    """

    FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    COLORS = (45, 39, 75, 105, 141, 177, 213, 177, 141, 105, 75, 39)  # cyan -> violet -> pink and back (256-color)

    def __init__(self, msg=""):
        self.msg = msg
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="raps-splash")
        self._lock = threading.Lock()
        self.running = False

    def start(self):
        self.running = True
        sys.stderr.write("\033[?25l")  # hide the cursor
        self._thread.start()
        return self

    def set(self, msg):
        self.msg = msg

    def frame(self, i):
        n = len(self.COLORS)
        col = lambda k: f"\033[38;5;{self.COLORS[k % n]}m"  # noqa: E731
        title = "".join(f"{col(i // 2 + k)}{c}" for k, c in enumerate("raps live"))
        dots = "".join("●" if (i // 3) % 4 > k else "○" for k in range(3))
        return (f"\r\033[K {col(i // 2)}\033[1m{self.FRAMES[i % len(self.FRAMES)]}\033[0m "
                f"\033[1m{title}\033[0m  \033[38;5;245m{self.msg} \033[38;5;240m{dots}\033[0m")

    def _run(self):
        i = 0
        while not self._stop.wait(0.08):
            with self._lock:
                if self._stop.is_set():
                    break
                sys.stderr.write(self.frame(i))
                sys.stderr.flush()
            i += 1

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._stop.set()
        with self._lock:  # waits for a frame being drawn
            sys.stderr.write("\r\033[K\033[?25h")  # clear the line, show the cursor
            sys.stderr.flush()


# `raps live` shows the splash before loading anything heavy (the console takes a while to start), but not while
# tab-completing or printing --help
_SPLASH = None
if (sys.argv[1:2] == ["live"] and sys.stderr.isatty() and "_ARGCOMPLETE" not in os.environ
        and not {"-h", "--help"} & set(sys.argv)):
    import atexit
    _SPLASH = Splash("starting").start()
    atexit.register(_SPLASH.stop)

# Implement shell completion using argcomplete
# Importing all of raps' dependencies like pandas etc can be rather slow, often taking 1-2 seconds. So for snappy shell
# completion we need avoid imports on the shell completion path. We could do this by shuffling the code around to
# create the parser without importing any heavy-weight libraries. But that would be a pain to maintain and track that
# pandas or scipy aren't accidentally imported transitively. Pandas can also be convenient to use in validating
# SimConfig etc, which is needed to build the argparser. So instead, we cache the generated argparser object so that
# shell completion can run without importing the rest of raps.
PARSER_CACHE = Path(__file__).parent / '.shell-completion-cache'


def shell_completion_add_parser(subparsers):
    parser = subparsers.add_parser("shell-completion", description=textwrap.dedent("""
        Register shell completion for RAPS.
    """).strip(), formatter_class=argparse.RawDescriptionHelpFormatter)

    # Run the command from argcomplete, this edits ~/.bash_completion to register argcomplete
    def impl(args):
        os.system("activate-global-python-argcomplete")

    parser.set_defaults(impl=impl)


def shell_complete():
    import argcomplete
    import dill
    try:
        parser = dill.loads(gzip.decompress(PARSER_CACHE.read_bytes()))
    except Exception:
        PARSER_CACHE.unlink(missing_ok=True)  # delete cache if corrupted somehow
        parser = argparse.ArgumentParser()
        # Use a dummy parser so that autocomplete still handles sys.exit tab complete if there's no
        # cache. Cache will be created on first run of `main.py`

    argcomplete.autocomplete(parser, always_complete_options=False)


def cache_parser(parser: argparse.ArgumentParser):
    import dill
    parser = copy.deepcopy(parser)
    subparsers = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    # Don't need to pickle the impl functions
    for subparser in subparsers.choices.values():
        subparser.set_defaults(impl=lambda args: None)

    pickled = gzip.compress(dill.dumps(parser), compresslevel=4, mtime=0)
    if not PARSER_CACHE.exists() or PARSER_CACHE.read_bytes() != pickled:
        try:  # Ignore if there's some kind of write or permission error
            PARSER_CACHE.write_bytes(pickled)
        except Exception:
            pass


def main(cli_args: list[str] | None = None):
    argv = sys.argv[1:] if cli_args is None else cli_args
    if argv[:1] == ["live"] and "_ARGCOMPLETE" not in os.environ:
        # Fast path: the console needs none of the simulator's parsers or imports (pandas, scipy, ...),
        # which alone take seconds to load. Say so right away, as the rest of start-up is still slow.
        if _SPLASH is not None:
            _SPLASH.set("loading raps")  # `import raps` loads the whole simulator engine
        try:
            import raps.live
            raps.live.splash = _SPLASH  # the steps in app.py report through raps.live.say
            parser = argparse.ArgumentParser(prog="raps", allow_abbrev=False)
            raps.live.live_add_parser(parser.add_subparsers(required=True))
            args = parser.parse_args(argv)
            return args.impl(args)
        finally:
            if _SPLASH is not None:
                _SPLASH.stop()  # also when something fails, so the traceback is not drawn over

    shell_complete()  # will output shell completion and sys.exit during tab complete

    from raps.helpers import check_python_version
    check_python_version()

    from raps.run_sim import run_sim_add_parser, run_parts_sim_add_parser, show_add_parser
    from raps.workloads import run_workload_add_parser
    from raps.telemetry import run_telemetry_add_parser, run_download_add_parser
    from raps.train_rl import train_rl_add_parser
    from raps.runs import runs_add_parser
    from raps.live import live_add_parser

    parser = argparse.ArgumentParser(
        description="""
            ExaDigiT Resource Allocator & Power Simulator (RAPS)
        """,
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(required=True)

    run_sim_add_parser(subparsers)
    run_parts_sim_add_parser(subparsers)
    show_add_parser(subparsers)
    run_workload_add_parser(subparsers)
    run_telemetry_add_parser(subparsers)
    run_download_add_parser(subparsers)
    train_rl_add_parser(subparsers)
    runs_add_parser(subparsers)
    live_add_parser(subparsers)
    shell_completion_add_parser(subparsers)

    cache_parser(parser)

    args = parser.parse_args(cli_args)
    assert args.impl, "subparsers should add an impl function to args"
    args.impl(args)


if __name__ == "__main__":
    main()
