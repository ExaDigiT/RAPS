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
Instructions for setup and usage are given below. 

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

    # Marconi100
    python main.py --system marconi100 -f ~/data/job_table.parquet 

    Note, once the data has been processed, it will be saved as an NPZ file, which
    can be more quickly started in subsequent simulations

    python main.py -f jobs_2024-02-20_12-20-39.npz

## Job-level power output example for replay of single job

    python main.py -f $DPATH/slurm/joblive/$DATEDIR $DPATH/jobprofile/$DATEDIR --jid 1234567 -o

## Compute stats on telemetry data, e.g., average job arrival time

    python -m raps.telemetry -f $DPATH/slurm/joblive/$DATEDIR $DPATH/jobprofile/jobprofile/$DATEDIR

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
