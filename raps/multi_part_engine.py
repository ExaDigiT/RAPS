from collections.abc import Iterable
from raps.engine import Engine, TickData
from raps.sim_config import MultiPartSimConfig


class MultiPartEngine:
    def __init__(self, sim_config: MultiPartSimConfig):
        if sim_config.replay:
            root_systems = set(s.system_name.split("/")[0] for s in sim_config.system_configs)
            # TODO should consider how to pass separate replay values for separate systems
            if len(root_systems) > 1:
                raise ValueError("Replay for multi-system runs is not supported")

        engines: dict[str, Engine] = {}

        for partition in sim_config.system_configs:
            engine = Engine(sim_config, partition=partition.system_name)
            engines[partition.system_name] = engine

        total_initial_jobs = sum(len(e.jobs) for e in engines.values())
        for engine in engines.values():
            engine.total_initial_jobs = total_initial_jobs

        self.partition_names = sorted(engines.keys())
        self.engines = engines
        first_engine = list(engines.values())[0]
        self.start = first_engine.start
        self.timestep_start = first_engine.timestep_start
        self.timestep_end = first_engine.timestep_end

    def run_simulation(self) -> Iterable[dict[str, TickData | None]]:
        generators = []
        for part in self.partition_names:
            generators.append(self.engines[part].run_simulation())
        for tick_datas in zip(*generators, strict=True):
            yield dict(zip(self.partition_names, tick_datas))

        # TODO need to add a mode to run the partitions in parallel
