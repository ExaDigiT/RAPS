""" Shortest-job first (SJF) job schedule simulator """

import json
import numpy as np
import random
import pandas as pd
import os
import time

from raps.helpers import check_python_version
check_python_version()

from raps.config import ConfigManager
from raps.constants import OUTPUT_PATH, SEED
from raps.cooling import ThermoFluidsModel
from raps.ui import LayoutManager
from raps.flops import FLOPSManager
from raps.plotting import Plotter
from raps.power import PowerManager, compute_node_power, compute_node_power_validate
from raps.power import compute_node_power_uncertainties, compute_node_power_validate_uncertainties
from raps.engine import Engine
from raps.job import Job
from raps.telemetry import Telemetry
from raps.workload import Workload
from raps.account import Accounts
from raps.weather import Weather
from raps.utils import create_casename, convert_to_seconds, write_dict_to_file, next_arrival_byconfargs
from raps.stats import get_engine_stats, get_job_stats, get_scheduler_stats
from raps.utils import convert_numpy_to_builtin

from args import args, args_dict


config = ConfigManager(system_name=args.system).get_config()

if args.seed:
    random.seed(SEED)
    np.random.seed(SEED)

if args.cooling:
    cooling_model = ThermoFluidsModel(**config)
    cooling_model.initialize()
    args.layout = "layout2"

    if args_dict['start']:
        cooling_model.weather = Weather(args_dict['start'], config=config)
else:
    cooling_model = None

if args.validate:
    if args.uncertainties:
        power_manager = PowerManager(compute_node_power_validate_uncertainties, **config)
    else:
        power_manager = PowerManager(compute_node_power_validate, **config)
else:
    if args.uncertainties:
        power_manager = PowerManager(compute_node_power_uncertainties, **config)
    else:
        power_manager = PowerManager(compute_node_power, **config)
args_dict['config'] = config
flops_manager = FLOPSManager(**args_dict)


if args.replay:

    td = Telemetry(**args_dict)
    jobs, timestep_start, timestep_end, args_from_file = td.load_jobs_times_args_from_files(files=args.replay, args=args)
    # TODO: Merge args and args_from_files? see telemetry.py:97

else:  # Synthetic jobs
    wl = Workload(config)
    jobs = getattr(wl, args.workload)(args=args)

    if args.verbose:
        for job_vector in jobs:
            job = Job(job_vector)
            print('jobid:', job.id, '\tlen(gpu_trace):', len(job.gpu_trace), '\twall_time(s):', job.wall_time)
        time.sleep(2)

    timestep_start = 0
    if hasattr(jobs[0],'end_time'):
        timestep_end = max([job.end_time for job in jobs])
    else:
        timestep_end = 88200  # 24 hours

    td = Telemetry(**args_dict)
    td.save_snapshot(jobs=jobs, timestep_start=timestep_start, timestep_end=timestep_end, args=args, filename=td.dirname)

if args.fastforward:
    args.fastforward = convert_to_seconds(args.fastforward)
    timestep_start = args.fastforward

if args.time:
    timestep_end = convert_to_seconds(args.time)


sc = Engine(
    power_manager=power_manager,
    flops_manager=flops_manager,
    cooling_model=cooling_model,
    jobs=jobs,
    **args_dict,
)

DIR_NAME = td.dirname
OPATH = OUTPUT_PATH / DIR_NAME
print("Output directory is: ", OPATH)
sc.opath = OPATH

if args.accounts:
    job_accounts = Accounts(jobs)
    if args.accounts_json:
        loaded_accounts = Accounts.from_json_filename(args.accounts_json)
        accounts = Accounts.merge(loaded_accounts, job_accounts)
    else:
        accounts = job_accounts
    sc.accounts = accounts

if args.plot or args.output:
    try:
        os.makedirs(OPATH)
    except OSError as error:
        print(f"Error creating directory: {error}")

if args.verbose:
    print(jobs)

total_timesteps = timestep_end - timestep_start
print(f'Simulating {len(jobs)} jobs for {total_timesteps} seconds from {timestep_start} to {timestep_end}.')
layout_manager = LayoutManager(args.layout, engine=sc, debug=args.debug, total_timesteps=total_timesteps, **config)
layout_manager.run(jobs, timestep_start=timestep_start, timestep_end=timestep_end)

engine_stats = get_engine_stats(sc)
job_stats = get_job_stats(sc)
scheduler_stats = get_scheduler_stats(sc)
# Following b/c we get the following error when we use PM100 telemetry dataset
# TypeError: Object of type int64 is not JSON serializable
try:
    print(json.dumps(engine_stats, indent=4))
    print(json.dumps(job_stats, indent=4))
    print(json.dumps(scheduler_stats, indent=4))
except:
    print(engine_stats)
    print(job_stats)
    print(scheduler_stats)


if args.plot:
    if 'power' in args.plot:
        pl = Plotter('Time (s)', 'Power (kW)', 'Power History', \
                     OPATH / f'power.{args.imtype}', \
                     uncertainties=args.uncertainties)
        x, y = zip(*power_manager.history)
        pl.plot_history(x, y)

    if 'util' in args.plot:
        pl = Plotter('Time (s)', 'System Utilization (%)', \
                     'System Utilization History', OPATH / f'util.{args.imtype}')
        x, y = zip(*sc.sys_util_history)
        pl.plot_history(x, y)

    if 'loss' in args.plot:
        pl = Plotter('Time (s)', 'Power Losses (kW)', 'Power Loss History', \
                     OPATH / f'loss.{args.imtype}', \
                     uncertainties=args.uncertainties)
        x, y = zip(*power_manager.loss_history)
        pl.plot_history(x, y)

        pl = Plotter('Time (s)', 'Power Losses (%)', 'Power Loss History', \
                     OPATH / f'loss_pct.{args.imtype}', \
                     uncertainties=args.uncertainties)
        x, y = zip(*power_manager.loss_history_percentage)
        pl.plot_history(x, y)

    if 'pue' in args.plot:
        if cooling_model:
            ylabel = 'pue'
            title = 'FMU ' + ylabel + 'History'
            pl = Plotter('Time (s)', ylabel, title, OPATH / f'pue.{args.imtype}', \
                         uncertainties=args.uncertainties)
            df = pd.DataFrame(cooling_model.fmu_history)
            df.to_parquet('cooling_model.parquet', engine='pyarrow')
            pl.plot_history(df['time'], df[ylabel])
        else:
            print('Cooling model not enabled... skipping output of plot')

    if 'temp' in args.plot:
        if cooling_model:
            ylabel = 'Tr_pri_Out[1]'
            title = 'FMU ' + ylabel + 'History'
            pl = Plotter('Time (s)', ylabel, title, OPATH / 'temp.svg')
            df = pd.DataFrame(cooling_model.fmu_history)
            df.to_parquet('cooling_model.parquet', engine='pyarrow')
            pl.plot_compare(df['time'], df[ylabel])
        else:
            print('Cooling model not enabled... skipping output of plot')

if args.output:

    if args.uncertainties:
        # Parquet cannot handle annotated ufloat format AFAIK
        print('Data dump not implemented using uncertainties!')
    else:
        if cooling_model:
            df = pd.DataFrame(cooling_model.fmu_history)
            df.to_parquet(OPATH / 'cooling_model.parquet', engine='pyarrow')

        df = pd.DataFrame(power_manager.history)
        df.to_parquet(OPATH / 'power_history.parquet', engine='pyarrow')

        df = pd.DataFrame(power_manager.loss_history)
        df.to_parquet(OPATH / 'loss_history.parquet', engine='pyarrow')

        df = pd.DataFrame(sc.sys_util_history)
        df.to_parquet(OPATH / 'util.parquet', engine='pyarrow')

        # Schedule history
        job_history = pd.DataFrame(sc.get_job_history_dict())
        job_history.to_csv(OPATH / "job_history.csv", index=False)

        scheduler_running_history = pd.DataFrame(sc.get_scheduler_running_history())
        job_history.to_csv(OPATH / "running_history.csv", index=False)
        scheduler_queue_history = pd.DataFrame(sc.get_scheduler_running_history())
        job_history.to_csv(OPATH / "queue_history.csv", index=False)

        try:
            with open(OPATH / 'stats.out', 'w') as f:
                json.dump(engine_stats, f, indent=4)
                json.dump(job_stats, f, indent=4)
        except TypeError:  # Is this the correct error code?
            write_dict_to_file(engine_stats, OPATH / 'stats.out')
            write_dict_to_file(job_stats, OPATH / 'stats.out')

        if args.accounts:
            try:
                with open(OPATH / 'accounts.json', 'w') as f:
                    json_string = json.dumps(sc.accounts.to_dict())
                    f.write(json_string)
            except TypeError:
                write_dict_to_file(sc.accounts.to_dict(), OPATH / 'accounts.json')
    print("Output directory is: ", OPATH)  # If output is enabled, the user wants this information as last output
