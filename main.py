"""
ExaDigiT Resource Allocator & Power Simulator (RAPS)
"""
import yaml
import argparse
import sys
from pathlib import Path
from raps.helpers import check_python_version
from raps.sim_config import SimConfig
from raps.run_sim import run_sim, run_multi_part_sim
from raps.workload import run_workload
from raps.telemetry import run_telemetry, run_telemetry_add_args
from raps.utils import pydantic_add_args, yaml_dump
from pydantic_settings import SettingsConfigDict

check_python_version()


def read_sim_yaml(config_file: str):
    if config_file == "-":
        return yaml.safe_load(sys.stdin.read())
    elif config_file:
        return yaml.safe_load(Path(config_file).read_text())
    else:
        return {}


CLI_CONFIG = SettingsConfigDict(
    cli_implicit_flags=True,
    cli_kebab_case=True,
)


def main():
    parser = argparse.ArgumentParser(
        description="""
            ExaDigiT Resource Allocator & Power Simulator (RAPS)
        """,
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(required=True)

    # Shortcut for common sim args
    sim_shortcuts = {
        "partitions": "x",
        "cooling": "c",
        "simulate-network": "net",
        "fastforward": "ff",
        "time": "t",
        "debug": "d",
        "numjobs": "n",
        "verbose": "v",
        "output": "o",
        "uncertainties": "u",
        "plot": "p",
        "replay": "f",
        "workload": "w",
    }

    # ==== raps run ====
    cmd_run = subparsers.add_parser("run", description="""
        Run single-partition (homogeneous) systems. Supports synthetic workload generation or
        telemetry replay, dynamic power modeling (including conversion losses), and optional
        coupling to a thermo-fluids cooling model. Produces performance, utilization, and
        energy metrics, with optional plots and output files for analysis and validation.
    """)
    cmd_run.add_argument("config_file", nargs="?", default=None, help="""
        YAML sim config file, can be used to configure an experiment instead of using CLI
        flags. Pass "-" to read from stdin.
    """)
    cmd_run_validate = pydantic_add_args(cmd_run, SimConfig, model_config={
        **CLI_CONFIG,
        "cli_shortcuts": sim_shortcuts,
    })

    def cmd_run_func(args):
        sim_config = cmd_run_validate(args, read_sim_yaml(args.config_file))
        run_sim(sim_config)
    cmd_run.set_defaults(func=cmd_run_func)

    # ==== raps run-multi-part ====
    # It might make sense to combine these into a single entrypoint. Though the multi-part run
    # #doesn't support UI or the same output options.
    cmd_run_multi_part = subparsers.add_parser("run-multi-part", description="""
        Simulates multi-partition (heterogeneous) systems. Supports replaying telemetry or
        generating synthetic workloads across CPU-only, GPU, and mixed partitions. Initializes
        per-partition power, FLOPS, and scheduling models, then advances simulations in lockstep.
        Outputs per-partition performance, utilization, and energy statistics for systems such as
        MIT Supercloud, Setonix, Adastra, and LUMI.
    """)
    cmd_run_multi_part.add_argument("config_file", nargs="?", default=None, help="""
        YAML sim config file, can be used to configure an experiment instead of using CLI
        flags. Pass "-" to read from stdin.
    """)
    cmd_run_multi_part_validate = pydantic_add_args(cmd_run_multi_part, SimConfig, model_config={
        **CLI_CONFIG,
        "cli_shortcuts": sim_shortcuts,
    })

    def cmd_run_multi_part_func(args):
        sim_config = cmd_run_multi_part_validate(args, read_sim_yaml(args.config_file))
        run_multi_part_sim(sim_config)
    cmd_run_multi_part.set_defaults(func=cmd_run_multi_part_func)

    # ==== raps show ====
    cmd_show = subparsers.add_parser("show", description="""
        Outputs the given CLI args as a YAML config file that can be used to re-run the same
        simulation.
    """)
    cmd_show.add_argument("config_file", nargs="?", default=None, help="""
        Input YAML sim config file. Can be used to slightly modify an existing sim config.
    """)
    cmd_show.add_argument("--show-defaults", default=False, help="""
        If true, include defaults in the output YAML
    """)
    cmd_show_validate = pydantic_add_args(cmd_show, SimConfig, model_config={
        **CLI_CONFIG,
        "cli_shortcuts": sim_shortcuts,
    })

    def cmd_show_func(args):
        sim_config = cmd_show_validate(args, read_sim_yaml(args.config_file))
        sim_config = sim_config.model_dump(mode="json",
                                           exclude_defaults=not args.show_defaults)
        print(yaml_dump(sim_config), end="")
    cmd_show.set_defaults(func=cmd_show_func)

    # ==== raps workload ====
    # TODO: Separate the arguments for this command
    cmd_workload = subparsers.add_parser("workload", description="""
        Saves workload as a snapshot.
    """)
    cmd_workload.add_argument("config_file", nargs="?", default=None, help="""
        YAML sim config file, can be used to configure an experiment instead of using CLI
        flags. Pass "-" to read from stdin.
    """)
    cmd_workload_validate = pydantic_add_args(cmd_workload, SimConfig, model_config={
        **CLI_CONFIG,
        "cli_shortcuts": sim_shortcuts,
    })

    def cmd_workload_func(args):
        sim_config = cmd_workload_validate(args, read_sim_yaml(args.config_file))
        run_workload(sim_config)
    cmd_show.set_defaults(func=cmd_workload_func)

    # ==== raps telemetry ====
    cmd_telemetry = subparsers.add_parser("telemetry", description="""
        Telemetry data validator
    """)
    run_telemetry_add_args(cmd_telemetry)
    cmd_telemetry.set_defaults(func=run_telemetry)

    # TODO: move telemetry and other misc scripts into here

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
