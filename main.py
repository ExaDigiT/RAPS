#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
"""
ExaDigiT Resource Allocator & Power Simulator (RAPS)
"""
import argparse
from pathlib import Path
import os
import textwrap
import copy
import gzip
import dill
import argcomplete

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
    try:
        parser = dill.loads(gzip.decompress(PARSER_CACHE.read_bytes()))
    except Exception:
        PARSER_CACHE.unlink(missing_ok=True)  # delete cache if corrupted somehow
        parser = argparse.ArgumentParser()
        # Use a dummy parser so that autocomplete still handles sys.exit tab complete if there's no
        # cache. Cache will be created on first run of `main.py`

    argcomplete.autocomplete(parser, always_complete_options=False)


def cache_parser(parser: argparse.ArgumentParser):
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
    shell_complete()  # will output shell completion and sys.exit during tab complete

    from raps.helpers import check_python_version
    check_python_version()

    from raps.run_sim import run_sim_add_parser, run_parts_sim_add_parser, show_add_parser
    from raps.workloads import run_workload_add_parser
    from raps.telemetry import run_telemetry_add_parser, run_download_add_parser
    from raps.train_rl import train_rl_add_parser

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
    shell_completion_add_parser(subparsers)

    cache_parser(parser)

    args = parser.parse_args(cli_args)
    assert args.impl, "subparsers should add an impl function to args"
    args.impl(args)


if __name__ == "__main__":
    main()
