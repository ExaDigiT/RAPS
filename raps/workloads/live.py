import math
import numpy as np
from raps.sim_config import SingleSimConfig
from raps.telemetry import Telemetry
from raps.utils import create_file_indexed
from .utils import plot_job_hist

def continuous_job_generation(self, *, engine, timestep, jobs):
    # print("if len(engine.queue) <= engine.continuous_workload.args.maxqueue:")
    # print(f"if {len(engine.queue)} <= {engine.continuous_workload.args.maxqueue}:")
    if len(engine.queue) <= engine.continuous_workload.args.maxqueue:
        new_jobs = engine.continuous_workload.generate_jobs().jobs
        jobs.extend(new_jobs)


def run_workload(sim_config: SingleSimConfig):
    args = sim_config.get_legacy_args()
    args_dict = sim_config.get_legacy_args()
    config = sim_config.system_configs[0].get_legacy()

    if sim_config.replay:
        td = Telemetry(**args_dict)
        jobs = td.load_from_files(sim_config.replay).jobs
    else:
        workload = Workload(args, config)
        jobs = getattr(workload, sim_config.workload)(args=sim_config.get_legacy_args())
    plot_job_hist(jobs,
                  config=config,
                  dist_split=sim_config.multimodal,
                  gantt_nodes=sim_config.gantt_nodes)

    out = sim_config.get_output()
    if out:
        timestep_start = min([x.submit_time for x in jobs])
        timestep_end = math.ceil(max([x.submit_time for x in jobs]) + max([x.expected_run_time for x in jobs]))
        filename = create_file_indexed('wl', path=str(out), create=False, ending="npz").split(".npz")[0]
        # savez_compressed add npz itself, but create_file_indexed needs to check for .npz to find existing files
        np.savez_compressed(filename, jobs=jobs, timestep_start=timestep_start, timestep_end=timestep_end, args=args)
        print(filename + ".npz")  # To std-out to show which npz was created.
