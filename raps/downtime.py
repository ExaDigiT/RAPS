from __future__ import annotations
from typing import TYPE_CHECKING
from raps.job import JobState
import numpy as np


if TYPE_CHECKING:
    from raps.engine import Engine


class Downtime:

    def __init__(self, *,
                 first_downtime,
                 downtime_interval,
                 downtime_length,
                 debug=False
                 ):
        self.skip = False
        if downtime_length == 0 or downtime_interval == 0 or \
           downtime_length is None or downtime_interval is None:
            self.skip = True
        self.interval: int = downtime_interval
        self.length: int = downtime_length
        self.start: int = first_downtime
        self.end: int = 0
        self.down: bool = False
        self.debug = debug

    def check_and_trigger(self, *,
                          timestep: int,
                          engine: Engine
                          ):
        if self.skip:
            return False  # Dont simulate downtime
        if timestep > self.start and not self.down:
            self.simulate_down(engine=engine)
            this_downtime_length = np.random.normal(self.length, 30 * 60)  # 30 minutes std variance around the downtime
            self.end = timestep + this_downtime_length
            self.start = self.start + self.interval  # Next start
            return True  # System went down
        if timestep > self.end and self.down:
            self.simulate_up(engine=engine)
            return True  # System went up
        return False  # No change

    def simulate_down(self, *,
                      engine: Engine
                      ):
        if self.debug:
            print("Simulated downtime: before downtime start")
            print(f"Running: {len(engine.running)}, queued: {len(engine.queue)}")

        # engine.resource_manager.down_nodes.update(engine.resource_manager.nodes)  # down_nodes are a set
        # engine.resource_manager.available_nodes[:] = []

        for job in engine.running:
            job._state = JobState.CANCELLED
            engine.power_manager.set_idle(job.scheduled_nodes)
            engine.resource_manager.free_nodes_from_job(job)

        # add all available nodes to down set.
        engine.resource_manager.down_nodes.update(
            engine.resource_manager.available_nodes)
        # clear available nodes
        engine.resource_manager.available_nodes[:] = []

        engine.queue += engine.running
        engine.running = []
        if self.debug:
            print("Simulated downtime: after downtime start")
            print(f"Running: {len(engine.running)}, queued: {len(engine.queue)}")
        self.down = True

    def simulate_up(self, *,
                    engine: Engine
                    ):
        self.down = False
        engine.resource_manager.available_nodes[:] = [n['id']
                                                      for n in engine.resource_manager.nodes if not n['is_down']]
        engine.down_nodes  # Careful!
        # these are the down nodes not managed by the resouce manager but given to the engine!
        engine.resource_manager.down_nodes.clear()
        engine.resource_manager.down_nodes.update(engine.config["DOWN_NODES"])  # Orig.
