"""
Module containing the primary commands for use in the CLI. The simulation logic itself is kept in
Engine and MultiPartEngine so that it can be used programmatically such as in the simulation server.
These functions just handle rendering the terminal UI and outputting results to files etc.
"""
import json
import pandas as pd
import sys
import yaml
import warnings
from pathlib import Path
from raps.ui import LayoutManager
from raps.plotting import Plotter
from raps.engine import Engine
from raps.multi_part_engine import MultiPartEngine
from raps.utils import write_dict_to_file, pydantic_add_args, SubParsers, yaml_dump
from raps.stats import (
    get_engine_stats,
    get_job_stats,
    get_scheduler_stats,
    get_network_stats,
    print_formatted_report
)

from raps.sim_config import SimConfig


def read_yaml(config_file: str):
    if config_file == "-":
        return yaml.safe_load(sys.stdin.read())
    elif config_file:
        return yaml.safe_load(Path(config_file).read_text())
    else:
        return {}


shortcuts = {
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


def run_sim_add_parser(subparsers: SubParsers):
    parser = subparsers.add_parser("run", description="""
        Run single-partition (homogeneous) systems. Supports synthetic workload generation or
        telemetry replay, dynamic power modeling (including conversion losses), and optional
        coupling to a thermo-fluids cooling model. Produces performance, utilization, and
        energy metrics, with optional plots and output files for analysis and validation.
    """)
    parser.add_argument("config_file", nargs="?", default=None, help="""
        YAML sim config file, can be used to configure an experiment instead of using CLI
        flags. Pass "-" to read from stdin.
    """)
    model_validate = pydantic_add_args(parser, SimConfig, model_config={
        "cli_shortcuts": shortcuts,
    })
    parser.set_defaults(
        impl=lambda args: run_sim(model_validate(args, read_yaml(args.config_file)))
    )


def run_sim(sim_config: SimConfig):
    if sim_config.verbose or sim_config.debug:
        print(f"SimConfig: {sim_config.model_dump_json(indent=4)}")
    if len(sim_config.system_configs) > 1:
        print("Use run-parts to run multi-partition simulations")
        sys.exit(1)

    engine, workload_data, time_delta = Engine.from_sim_config(sim_config)

    out = sim_config.output
    if out:
        out.mkdir(parents=True)
        engine.telemetry.save_snapshot(
            dest=str(out),
            result=workload_data,
            args=sim_config,
        )
    jobs = workload_data.jobs
    timestep_start, timestep_end = workload_data.telemetry_start, workload_data.telemetry_end
    total_timesteps = timestep_end - timestep_start

    downscale = sim_config.downscale
    downscale_str = ""if downscale == 1 else f"/{downscale}"
    print(f"Simulating {len(jobs)} jobs for {total_timesteps}{downscale_str}"
          f" seconds from {timestep_start} to {timestep_end}.")
    print(f"Simulation time delta: {time_delta}{downscale_str} s,"
          f"Telemetry trace quanta: {jobs[0].trace_quanta}{downscale_str} s.")
    layout_manager = LayoutManager(
        sim_config.layout, engine=engine,
        debug=sim_config.debug, total_timesteps=total_timesteps,
        args_dict=sim_config.get_legacy_args_dict(), **sim_config.system_configs[0].get_legacy(),
    )
    layout_manager.run(
        jobs,
        timestep_start=timestep_start, timestep_end=timestep_end, time_delta=time_delta,
    )

    engine_stats = get_engine_stats(engine)
    job_stats = get_job_stats(engine)
    scheduler_stats = get_scheduler_stats(engine)
    if engine.simulate_network:
        network_stats = get_network_stats(engine)
    else:
        network_stats = None

    print_formatted_report(
        engine_stats=engine_stats,
        job_stats=job_stats,
        scheduler_stats=scheduler_stats,
        network_stats=network_stats,
    )

    if downscale_str:
        downscale_str = "1" + downscale_str

    if sim_config.plot:
        assert out  # SimConfig validation should check this
        if 'power' in sim_config.plot:
            pl = Plotter(f"Time ({downscale_str}s)", 'Power (kW)', 'Power History',
                         out / f'power.{sim_config.imtype}',
                         uncertainties=sim_config.uncertainties)
            x, y = zip(*engine.power_manager.history)
            pl.plot_history(x, y)

        if 'util' in sim_config.plot:
            pl = Plotter(f"Time ({downscale_str}s)", 'System Utilization (%)',
                         'System Utilization History', out / f'util.{sim_config.imtype}')
            x, y = zip(*engine.sys_util_history)
            pl.plot_history(x, y)

        if 'loss' in sim_config.plot:
            pl = Plotter(f"Time ({downscale_str}s)", 'Power Losses (kW)', 'Power Loss History',
                         out / f'loss.{sim_config.imtype}',
                         uncertainties=sim_config.uncertainties)
            x, y = zip(*engine.power_manager.loss_history)
            pl.plot_history(x, y)

            pl = Plotter(f"Time ({downscale_str}s)", 'Power Losses (%)', 'Power Loss History',
                         out / f'loss_pct.{sim_config.imtype}',
                         uncertainties=sim_config.uncertainties)
            x, y = zip(*engine.power_manager.loss_history_percentage)
            pl.plot_history(x, y)

        if 'pue' in sim_config.plot:
            if engine.cooling_model:
                ylabel = 'pue'
                title = 'FMU ' + ylabel + 'History'
                pl = Plotter(f"Time ({downscale_str}s)", ylabel, title,
                             out / f'pue.{sim_config.imtype}',
                             uncertainties=sim_config.uncertainties)
                df = pd.DataFrame(engine.cooling_model.fmu_history)
                df.to_parquet('cooling_model.parquet', engine='pyarrow')
                pl.plot_history(df['time'], df[ylabel])
            else:
                print('Cooling model not enabled... skipping output of plot')

        if 'temp' in sim_config.plot:
            if engine.cooling_model:
                ylabel = 'Tr_pri_Out[1]'
                title = 'FMU ' + ylabel + 'History'
                pl = Plotter(f"Time ({downscale_str}s)", ylabel, title, out / 'temp.svg')
                df = pd.DataFrame(engine.cooling_model.fmu_history)
                df.to_parquet('cooling_model.parquet', engine='pyarrow')
                pl.plot_compare(df['time'], df[ylabel])
            else:
                print('Cooling model not enabled... skipping output of plot')

    if out:
        if sim_config.uncertainties:
            # Parquet cannot handle annotated ufloat format AFAIK
            print('Data dump not implemented using uncertainties!')
        else:
            if engine.cooling_model:
                df = pd.DataFrame(engine.cooling_model.fmu_history)
                df.to_parquet(out / 'cooling_model.parquet', engine='pyarrow')

            df = pd.DataFrame(engine.power_manager.history)
            df.to_parquet(out / 'power_history.parquet', engine='pyarrow')

            df = pd.DataFrame(engine.power_manager.loss_history)
            df.to_parquet(out / 'loss_history.parquet', engine='pyarrow')

            df = pd.DataFrame(engine.sys_util_history)
            df.to_parquet(out / 'util.parquet', engine='pyarrow')

            # Schedule history
            job_history = pd.DataFrame(engine.get_job_history_dict())
            job_history.to_csv(out / "job_history.csv", index=False)

            scheduler_running_history = pd.DataFrame(engine.get_scheduler_running_history())
            scheduler_running_history.to_csv(out / "running_history.csv", index=False)
            scheduler_queue_history = pd.DataFrame(engine.get_scheduler_running_history())
            scheduler_queue_history.to_csv(out / "queue_history.csv", index=False)

            try:
                with open(out / 'stats.out', 'w') as f:
                    json.dump(engine_stats, f, indent=4)
                    json.dump(job_stats, f, indent=4)
            except TypeError:  # Is this the correct error code?
                write_dict_to_file(engine_stats, out / 'stats.out')
                write_dict_to_file(job_stats, out / 'stats.out')

            if sim_config.accounts:
                try:
                    with open(out / 'accounts.json', 'w') as f:
                        json_string = json.dumps(engine.accounts.to_dict())
                        f.write(json_string)
                except TypeError:
                    write_dict_to_file(engine.accounts.to_dict(), out / 'accounts.json')
        print("Output directory is: ", out)  # If output is enabled, the user wants this information as last output


def run_parts_sim_add_parser(subparsers: SubParsers):
    parser = subparsers.add_parser("run-parts", description="""
        Simulates multi-partition (heterogeneous) systems. Supports replaying telemetry or
        generating synthetic workloads across CPU-only, GPU, and mixed partitions. Initializes
        per-partition power, FLOPS, and scheduling models, then advances simulations in lockstep.
        Outputs per-partition performance, utilization, and energy statistics for systems such as
        MIT Supercloud, Setonix, Adastra, and LUMI.
    """)
    parser.add_argument("config_file", nargs="?", default=None, help="""
        YAML sim config file, can be used to configure an experiment instead of using CLI
        flags. Pass "-" to read from stdin.
    """)
    model_validate = pydantic_add_args(parser, SimConfig, model_config={
        "cli_shortcuts": shortcuts,
    })
    parser.set_defaults(
        impl=lambda args: run_parts_sim(model_validate(args, read_yaml(args.config_file)))
    )


def run_parts_sim(sim_config: SimConfig):

    if len(sim_config.system_configs) == 1:
        warnings.warn(
            "run_parts_sim is usually for multiple partitions. Did you mean to run with one?",
            UserWarning
        )

    multi_engine, workload_results, timestep_start, timestep_end, time_delta = \
        MultiPartEngine.from_sim_config(sim_config)

    if sim_config.output:
        for part, engine in multi_engine.engines.items():
            engine.telemetry.save_snapshot(
                dest=str(sim_config.output / part.split('/')[-1]),
                result=workload_results[part],
                args=sim_config,
            )
    jobs = {p: w.jobs for p, w in workload_results.items()}

    ui_update_freq = sim_config.system_configs[0].scheduler.ui_update_freq
    gen = multi_engine.run_simulation(jobs, timestep_start, timestep_end, time_delta)

    for tick_datas in gen:
        sys_power = 0
        tick_datas = {k: v for k, v in tick_datas.items() if v}  # Filter nones
        timestep = list(tick_datas.values())[0].current_timestep if tick_datas else None

        if timestep and timestep % ui_update_freq == 0:
            for part, tick_data in tick_datas.items():
                engine = multi_engine.engines[part]

                sys_util = engine.sys_util_history[-1] if engine.sys_util_history else (0, 0.0)
                if hasattr(engine.resource_manager, 'allocated_cpu_cores'):
                    allocated_cores = engine.resource_manager.allocated_cpu_cores
                    print(
                        f"[DEBUG] {part} - Timestep {timestep} - Jobs running: {len(engine.running)} -",
                        f"Utilization: {sys_util[1]:.2f}% - Allocated Cores: {allocated_cores} - ",
                        f"Power: {engine.sys_power:.1f}kW",
                        flush=True,
                    )
                sys_power += engine.sys_power
            print(f"system power: {sys_power:.1f}kW", flush=True)

    print("Simulation complete.", flush=True)

    # Print statistics for each partition
    for part, engine in multi_engine.engines.items():
        print(f"\n=== Partition: {part} ===")

        engine_stats = get_engine_stats(engine)
        job_stats = get_job_stats(engine)
        scheduler_stats = get_scheduler_stats(engine)
        network_stats = get_network_stats(engine) if sim_config.simulate_network else None

        # Print a formatted report
        print_formatted_report(
            engine_stats=engine_stats,
            job_stats=job_stats,
            scheduler_stats=scheduler_stats,
            network_stats=network_stats,
        )


def show_add_parser(subparsers: SubParsers):
    parser = subparsers.add_parser("show", description="""
        Outputs the given CLI args as a YAML config file that can be used to re-run the same
        simulation.
    """)
    parser.add_argument("config_file", nargs="?", default=None, help="""
        Input YAML sim config file. Can be used to slightly modify an existing sim config.
    """)
    parser.add_argument("--show-defaults", default=False, help="""
        If true, include defaults in the output YAML
    """)
    model_validate = pydantic_add_args(parser, SimConfig, model_config={
        "cli_shortcuts": shortcuts,
    })

    def impl(args):
        sim_config = model_validate(args, read_yaml(args.config_file))
        show(sim_config, show_defaults=args.show_defaults)

    parser.set_defaults(impl=impl)


def show(sim_config: SimConfig, show_defaults=False):
    data = sim_config.model_dump(mode="json", exclude_defaults=not show_defaults)
    print(yaml_dump(data), end="")
