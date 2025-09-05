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

Note: Requires python3.12 or greater.

    pip install -e .

## Usage and help menu

    raps run -h

## Run simulator with default synthetic workload

    raps run

## Run simulator with telemetry replay

    # Frontier
    DATEDIR="date=2024-01-18"
    DPATH=~/data/frontier-sample-2024-01-18
    raps run -f $DPATH/slurm/joblive/$DATEDIR,$DPATH/jobprofile/$DATEDIR

## Open Telemetry dataset

For Marconi supercomputer, download `job_table.parquet` from https://zenodo.org/records/10127767

    # Marconi100
    raps run --system marconi100 -f ~/data/marconi100/job_table.parquet

For Adastra MI250 supercomputer, download 'AdastaJobsMI250_15days.parquet' from https://zenodo.org/records/14007065

    # Adastra MI250
    raps run --system adastraMI250 -f AdastaJobsMI250_15days.parquet

For Google cluster trace v2

    raps run --system gcloudv2 -f ~/data/gcloud/v2/google_cluster_data_2011_sample --ff 600

    # analyze dataset
    raps telemetry --system gcloudv2 -f ~/data/gcloud/v2/google_cluster_data_2011_sample -v

For MIT Supercloud

    # Following is the directory that contains slurm-log.csv and cpu and gpu directories
    DPATH=/path/to/mit/data

    # Download the dataset - note the first time will build a file-manifest.txt file with all the files on S3
    # this will take some time, but subsequent calls should be much faster.
    # Also, this command will dump output to `source_data` directory, or can specify directory using `--outdir`
    python -m raps.dataloaders.mit_supercloud.cli download --start 2021-05-21T13:00 --end 2021-05-21T14:00

    # Load data and run simulation - will save data as part-cpu.npz and part-gpu.npz files
    raps run-parts -x mit_supercloud -f $DPATH --system mit_supercloud --start 2021-05-21T13:00 --end 2021-05-21T14:00
    # Note: if no start, end dates provided will default to run 24 hours between
    # 2021-05-21T00:00 to 2021-05-22T00:00 set by defaults in raps/dataloaders/mit_supercloud/utils.py

    # Re-run simulation using npz files (much faster load)
    raps run-parts -x mit_supercloud -f part-*.npz --system mit_supercloud

    # Synthetic tests for verification studies:
    raps run-parts -x mit_supercloud -w multitenant

For Lumi

    # Synthetic test for Lumi:
    raps run-parts -x lumi

## Perform Network Simulation

Lassen is one of the few datasets that has networking data. See `raps/dataloaders/lassen.py` for how to
get the datasets. To run a network simulation, use the following command:

    raps run -f ~/data/lassen/Lassen-Supercomputer-Job-Dataset --system lassen --policy fcfs --backfill firstfit --ff 365d -t 12h --arrival poisson --net

## Snapshot of extracted workload data

To reduce the expense of extracting the needed data from the telemetry parquet files,
RAPS saves a snapshot of the extracted data in NPZ format. The NPZ file can be
given instead of the parquet files for more quickly running subsequent simulations, e.g.:

    raps run -f jobs_2024-02-20_12-20-39.npz

## Cooling models

We provide several cooling models in the repo https://code.ornl.gov/exadigit/POWER9CSM

    git submodule update --init --recursive

Will install the POWER9CSM in the models folder. To activate cooling when running RAPS,
use `--cooling` or `-c` argument. e.g.,

    raps run --system marconi100 -c

    raps run --system lassen -c

    raps run --system summit -c

## Support for multiple system partitions

Multi-partition systems are supported by running `raps multi-parts ...` command, where a list of partitions can be specified using the `-x` flag as follows:

    raps run-parts -x setonix/part-cpu setonix/part-gpu

or simply:

    raps run-parts -x setonix

This will simulate synthetic workloads on two partitions as defined in `config/setonix-cpu` and `config/setonix-gpu`. To replay telemetry workloads from another system, e.g., Marconi100's PM100 dataset, first create a .npz snapshot of the telemetry data, e.g.,

    raps run-parts --system marconi100 -f /path/to/marconi100/job_table.parquet

This will dump a .npz file with a randomized name, e.g. ac23db.npz. Let's rename this file to pm100.npz for clarity. Note: can control-C when the simulation starts. Now, this pm100.npz file can be used as follows:

    raps run-parts -x setonix -f pm100.npz --arrival poisson --scale 192

## Modifications to telemetry replay

There are three ways to modify replaying of telemetry data:

1. `--arrival`. Changing the arrival time distribution - replay cases will default to `--arrival prescribed`, where the jobs will be submitted exactly as they were submitted on the physical machine. This can be changed to `--arrival poisson` to change when the jobs arrive, which is especially useful in cases where there may be gaps in time, e.g., when the system goes down for several days, or the system is is underutilized.
python main.py -f $DPATH/slurm/joblive/$DATEDIR,$DPATH/jobprofile/$DATEDIR --arrival poisson

2. `--policy`. Changing the way the jobs are scheduled. The `--policy` flag will be set by default to `replay` in cases where a telemetry file is provided, in which case the jobs will be scheduled according to the start times provided. Changing the `--policy` to `fcfs` or `backfill` will use the internal scheduler, e.g.:

    python main.py -f $DPATH/slurm/joblive/$DATEDIR,$DPATH/jobprofile/$DATEDIR --policy fcfs --backfill firstfit -t 12h

3. `--scale`. Changing the scale of each job in the telemetry data. The `--scale` flag will specify the maximum number of nodes for each job (generally set this to the max number of nodes of the smallest partition), and randomly select the number of nodes for each job from one to max nodes. This flag is useful when replaying telemetry from a larger system onto a smaller system.

4. `--shuffle`. Shuffle the jobs before playing.

## Job-level power output example for replay of single job

    raps run -f $DPATH/slurm/joblive/$DATEDIR,$DPATH/jobprofile/$DATEDIR --jid 1234567 -o

## Compute stats on telemetry data, e.g., average job arrival time

    raps telemetry -f $DPATH/slurm/joblive/$DATEDIR,$DPATH/jobprofile/$DATEDIR

## Build and run Docker container

    make docker_build && make docker_run

## Third party schedulers

To install third-party schedulers, such as ScheduleFlow, run:

    git submodule update --init --recursive

### Setup Simulation Server

See instructions in [server/README.md](https://code.ornl.gov/exadigit/simulationserver)

### Setup Dashboard

See instructions in [dashboard/README.md](https://code.ornl.gov/exadigit/simulation-dashboard)

## Running Tests

RAPS uses [pytest](https://docs.pytest.org/) for its test suite.  
Before running tests, ensure that you have a valid data directory available (e.g., `/opt/data`) and set the environment variable `RAPS_DATA_DIR` to point to it.

### Run all tests
```bash
RAPS_DATA_DIR=/opt/data pytest -n auto -x
```

By default, tests are parallelized with `pytest-xdist` (`-n auto`) to speed up execution.
The `-x` flag stops execution after the first failure. Add `-v` to run in verbose mode.

### Run tests on multi-partition systems

```bash
pytest -v -k "multi_part_sim"
```

### Run only network-related tests

```bash
RAPS_DATA_DIR=/opt/data pytest -n auto -x -m network
```

See `pytest.ini` for the different options for `-m`.

### Run a specific test file

```bash
RAPS_DATA_DIR=/opt/data pytest tests/systems/test_engine.py
```

### Contributing Code

Install pre-commit hooks as set by the project:
```
pip install pre-commit
pre-commit install
```

## Authors

Many thanks to the contributors of ExaDigiT/RAPS.
The full list of contributors and organizations involved are found in CONTRIBUTORS.txt.

## Citation

If you use ExaDigiT or RAPS in your research, please cite our work:

    @inproceedings{inproceedings,
      title={A Digital Twin Framework for Liquid-cooled Supercomputers as Demonstrated at Exascale},
      author={Brewer, Wesley and Maiterth, Matthias and Kumar, Vineet and Wojda, Rafal and Bouknight, Sedrick and Hines, Jesse and Shin, Woong and Greenwood, Scott and Grant, David and Williams, Wesley and Wang, Feiyi},
      booktitle={SC24: International Conference for High Performance Computing, Networking, Storage and Analysis},
      pages={1--18},
      year={2024},
      organization={IEEE}
    }

    @misc{doecode_127899,
      title = {ExaDigiT/RAPS},
      author = {Brewer, Wesley and Maiterth, Matthias and Bouknight, Sedrick and Hines, Jesse and Webb, Tyler J.},
      doi = {10.11578/dc.20240627.4},
      url = {https://doi.org/10.11578/dc.20240627.4},
      howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20240627.4}},
      year = {2024},
      month = {jun}
    }

Thank you for your support!

## License

ExaDigiT/RAPS is distributed under the terms of both the MIT license and the Apache License (Version 2.0).
Users may choose either license, at their option.

All new contributions must be made under both the MIT and Apache-2.0 licenses.
See LICENSE-MIT, LICENSE-APACHE, COPYRIGHT, NOTICE, and CONTRIBUTORS.txt for details.

SPDX-License-Identifier: (Apache-2.0 OR MIT)

## Attributions

Map data used in this project is provided by [OpenStreetMap](https://www.openstreetmap.org/copyright) and is available under the Open Database License (ODbL). © OpenStreetMap contributors.

Weather data used in this project is provided by the [Open-Meteo API](https://open-meteo.com/en/docs). Open-Meteo offers free weather forecast data for various applications, and their API provides easy access to weather information without requiring user authentication.
