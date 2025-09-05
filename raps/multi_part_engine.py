from collections.abc import Iterable
from raps.engine import Engine, TickData
from raps.sim_config import SimConfig
from raps.utils import WorkloadData


class MultiPartEngine:
    def __init__(self, engines: dict[str, Engine], jobs: dict[str, list]):
        self.partition_names = sorted(engines.keys())
        self.engines = engines
        self.jobs = jobs

    @staticmethod
    def from_sim_config(sim_config: SimConfig):
        if sim_config.replay:
            root_systems = set(s.system_name.split("/")[0] for s in sim_config.system_configs)
            # TODO should consider how to pass separate replay values for separate systems
            if len(root_systems) > 1:
                raise ValueError("Replay for multi-system runs is not supported")

        workloads_by_partition: dict[str, WorkloadData] = {}
        engines: dict[str, Engine] = {}

        time_delta = 0
        for partition in sim_config.system_configs:
            name = partition.system_name
            engine, workload_data, time_delta = Engine.from_sim_config(
                sim_config, partition=name,
            )
            for job in workload_data.jobs:
                job.partition = name
            workloads_by_partition[name] = workload_data
            engines[name] = engine
        timestep_start = min(w.telemetry_start for w in workloads_by_partition.values())
        timestep_end = min(w.telemetry_end for w in workloads_by_partition.values())

        total_initial_jobs = sum(len(j.jobs) for j in workloads_by_partition.values())
        for engine in engines.values():
            engine.total_initial_jobs = total_initial_jobs

        multi_engine = MultiPartEngine(
            engines=engines,
            jobs={p: w.jobs for p, w in workloads_by_partition.items()},
        )

        return multi_engine, workloads_by_partition, timestep_start, timestep_end, time_delta

    def run_simulation(self, jobs: dict, timestep_start, timestep_end, time_delta=1
                       ) -> Iterable[dict[str, TickData | None]]:
        generators = []
        for part in self.partition_names:
            generators.append(self.engines[part].run_simulation(
                jobs[part], timestep_start, timestep_end, time_delta,
            ))
        for tick_datas in zip(*generators, strict=True):
            yield dict(zip(self.partition_names, tick_datas))

        # TODO need to add a mode to run the partitions in parallel
