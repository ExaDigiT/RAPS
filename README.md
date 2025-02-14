# ExaDigiT/RAPS

ExaDigiT's Resource Allocator and Power Simulator (RAPS) schedules workloads and 
estimates dynamic system power at specified time intervals. RAPS either schedules 
synthetic workloads or replays system telemetry workloads,
provides system monitoring during simulation, and an outputs a report of scheduling
and power statistics at the end of the simulation. RAPS also can interface with 
the FMU cooling model by providing CDU-level power inputs to the cooling model,
and reporting the statistics back to the user. RAPS also has built-in plotting
capabilities to generate plots of power and cooling at the end of simulation runs.
An optional RAPS dashboard is also provided, which requires also running the RAPS server.
Instructions for setup and usage are given below. An online documentation of ExaDigiT with a sub part concerning RAPS is also available [here](https://exadigit.readthedocs.io/en/latest/).

## Setup environment

Note: Requires python3.9 or greater.

    pip install -e .

## Usage and help menu

    python main.py -h

## Run simulator with default synthetic workload

    python main.py

## Run simulator with telemetry replay

    # Frontier 
    DATEDIR="date=2024-01-18"
    DPATH=~/data/frontier-sample-2024-01-18
    python main.py -f $DPATH/slurm/joblive/$DATEDIR $DPATH/jobprofile/$DATEDIR

## Open Telemetry dataset

For Marconi supercomputer, download `job_table.parquet` from https://zenodo.org/records/10127767

    # Marconi100
    python main.py --system marconi100 -f ~/data/marconi100/job_table.parquet 

For Adastra MI250 supercomputer, download 'AdastaJobsMI250_15days.parquet' from https://zenodo.org/records/14007065

    # Adastra MI250
    python main.py --system adastraMI250 -f AdastaJobsMI250_15days.parquet 

## Snapshot of extracted workload data

To reduce the expense of extracting the needed data from the telemetry parquet files,
RAPS saves a snapshot of the extracted data in NPZ format. The NPZ file can be
given instead of the parquet files for more quickly running subsequent simulations, e.g.:

    python main.py -f jobs_2024-02-20_12-20-39.npz

## Support for multiple system partitions

Multi-partition systems are supported by running the `multi-part-sim.py` script, where a list of configurations can be specified using the `-x` flag as follows:

    python multi-part-sim.py -x setonix/part-cpu setonix/part-gpu

or simply:

    python multi-part-sim.py -x setonix/* # bash

    python multi-part-sim.py -x 'setonix/*' # zsh

This will simulate synthetic workloads on two partitions as defined in `config/setonix-cpu` and `config/setonix-gpu`. To replay telemetry workloads from another system, e.g., Marconi100's PM100 dataset, first create a .npz snapshot of the telemetry data, e.g., 

    python main.py --system marconi100 -f /path/to/marconi100/job_table.parquet

This will dump a .npz file with a randomized name, e.g. ac23db.npz. Let's rename this file to pm100.npz for clarity. Note: can control-C when the simulation starts. Now, this pm100.npz file can be used with `multi-part-sim.py` as follows:

    python multi-part-sim.py -x setonix/* -f pm100.npz --arrival poisson --scale 192

## Modifications to telemetry replay

There are three ways to modify replaying of telemetry data:

    1. `--arrival`. Changing the arrival time distribution - replay cases will default to `--arrival prescribed`, where the jobs will be submitted exactly as they were submitted on the physical machine. This can be changed to `--arrival poisson` to change when the jobs arrive, which is especially useful in cases where there may be gaps in time, e.g., when the system goes down for several days, or the system is is underutilized.

    2. `--policy`. Changing the way the jobs are scheduled. The `--policy` flag will be set by default to `replay` in cases where a telemetry file is provided, in which case the jobs will be scheduled according to the start times provided. Changing the `--policy` to `fcfs` or `backfill` will use the internal scheduler.

    3. `--scale`. Changing the scale of each job in the telemetry data. The `--scale` flag will specify the maximum number of nodes for each job (generally set this to the max number of nodes of the smallest partition), and randomly select the number of nodes for each job from one to max nodes. This flag is useful when replaying telemetry from a larger system onto a smaller system.

    4. `--shuffle`. Shuffle the jobs before playing.

## Job-level power output example for replay of single job

    python main.py -f $DPATH/slurm/joblive/$DATEDIR $DPATH/jobprofile/$DATEDIR --jid 1234567 -o

## Compute stats on telemetry data, e.g., average job arrival time

    python -m raps.telemetry -f $DPATH/slurm/joblive/$DATEDIR $DPATH/jobprofile/$DATEDIR

## Build and run Docker container

    make docker_build && make docker_run

### Setup Simulation Server

See instructions in [server/README.md](https://code.ornl.gov/exadigit/simulationserver)

### Setup Dashboard

See instructions in [dashboard/README.md](https://code.ornl.gov/exadigit/simulation-dashboard)

## Authors:

Many thanks to the contributors of ExaDigiT/RAPS.  
The full list of contributors and organizations involved are found in CONTRIBUTORS.txt.  

## License:

ExaDigiT/RAPS is distributed under the terms of both the MIT license and the Apache License (Version 2.0).  
Users may choose either license, at their option.  

All new contributions must be made under both the MIT and Apache-2.0 licenses.  
See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, NOTICE, and CONTRIBUTORS.txt for details.  

SPDX-License-Identifier: (Apache-2.0 OR MIT)  

## Attributions

Map data used in this project is provided by [OpenStreetMap](https://www.openstreetmap.org/copyright) and is available under the Open Database License (ODbL). © OpenStreetMap contributors.

Weather data used in this project is provided by the [Open-Meteo API](https://open-meteo.com/en/docs). Open-Meteo offers free weather forecast data for various applications, and their API provides easy access to weather information without requiring user authentication.

