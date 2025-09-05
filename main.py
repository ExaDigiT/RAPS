"""
ExaDigiT Resource Allocator & Power Simulator (RAPS)
"""
import argparse
from raps.helpers import check_python_version
from raps.run_sim import run_sim_add_parser, run_parts_sim_add_parser, show_add_parser
from raps.workload import run_workload_add_parser
from raps.telemetry import run_telemetry_add_parser

check_python_version()


def main(cli_args: list[str] | None = None):
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

    # TODO: move other misc scripts into here

    args = parser.parse_args(cli_args)
    assert args.impl, "subparsers should add an impl function to args"
    args.impl(args)


if __name__ == "__main__":
    main()
