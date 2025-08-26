import gym
import numpy as np
from gym import spaces

from raps.engine import Engine
from raps.power import PowerManager, compute_node_power
from raps.flops import FLOPSManager
from raps.telemetry import Telemetry
from raps.workload import Workload
from raps.ui import LayoutManager


class RAPSEnv(gym.Env):
    """
    Minimal Gym-compatible wrapper around RAPS Engine
    for RL job scheduling experiments.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):
        super().__init__()
        # Store everything in self.args
        self.args_dict = kwargs  # dict
        self.cli_args = kwargs.get("args")  # Namespace
        self.config = kwargs.get("config")
        if self.cli_args is None:
            raise ValueError("RAPSEnv requires 'args' (argparse.Namespace) in kwargs")
        if self.config is None:
            raise ValueError("RAPSEnv requires 'config' in kwargs")

        # --- managers (minimal versions) ---
        self.power_manager = PowerManager(compute_node_power, **self.config)
        self.flops_manager = FLOPSManager(**self.args_dict)
        self.telemetry = Telemetry(**self.args_dict)

        # --- workload (synthetic for now) ---
        wl = Workload(self.cli_args, self.config)
        jobs = wl.generate_jobs()

        self.engine = Engine(
            power_manager=self.power_manager,
            flops_manager=self.flops_manager,
            jobs=jobs,
            **self.args_dict
        )

        self.layout_manager = LayoutManager(
            self.args_dict.get("layout"), engine=self.engine,
            debug=self.args_dict.get("debug", False),
            total_timesteps=self.args_dict.get("time", 1000),
            args_dict=self.args_dict,
            **self.config
        )

        # --- RL spaces ---
        max_jobs = 100
        job_features = 4  # [nodes, runtime, priority, wait_time]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(max_jobs, job_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(max_jobs)

    def reset(self, **kwargs):
        """Reset environment (new workload + engine)."""
        wl = Workload(self.cli_args, self.config)
        jobs = wl.generate_jobs()

        self.engine.jobs = jobs
        self.engine.timestep_start = 0
        # self.engine.timestep_end = int(max(job.wall_time for job in jobs))
        self.engine.timestep_end = 100
        self.engine.current_timestep = 0

        return self._get_state()

    def step(self, action):
        """
        Apply scheduling action.
        For now: action = index of job in queue to attempt scheduling.
        """
        # TODO: integrate action with real scheduling logic
        reward = np.random.rand()
        done = self.engine.current_timestep >= self.engine.timestep_end

        obs = self._get_state()

        # Compute info manually
        running_nodes = sum(getattr(j, "nodes_required", 0)
                            for j in self.engine.jobs
                            if getattr(j, "start_time", None) is not None)
        total_nodes = self.config.get("SC_NODES", 1)
        utilization = running_nodes / total_nodes

        info = {
            "utilization": utilization,
            "power": getattr(self.power_manager, "total_power", 0.0),
            "queue_length": len([j for j in self.engine.jobs if getattr(j, "start_time", None) is None]),
        }

        self.engine.current_timestep += 1
        return obs, reward, done, info

    def _get_state(self):
        """Construct simple state representation from engine's job queue."""
        # Example: take waiting jobs (haven’t started yet)
        job_queue = [j for j in self.engine.jobs if getattr(j, "start_time", None) is None]

        max_jobs, job_features = self.observation_space.shape
        state = np.zeros((max_jobs, job_features), dtype=np.float32)

        for i, job in enumerate(job_queue[:max_jobs]):
            features = [
                getattr(job, "nodes_required", 0),
                getattr(job, "wall_time", 0),
                getattr(job, "priority", 0),
                getattr(job, "wait_time", 0),  # may need to compute from current_timestep - qdt
            ]
            state[i, : len(features)] = features

        return state

    def render(self, mode="human"):
        print("Timestep:", self.engine.current_timestep,
              "Utilization:", self.telemetry.utilization(),
              "Power:", self.telemetry.power())
