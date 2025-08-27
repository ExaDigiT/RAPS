import gym
import numpy as np
from gym import spaces

from raps.engine import Engine
from raps.power import PowerManager, compute_node_power
from raps.flops import FLOPSManager
from raps.telemetry import Telemetry
from raps.workload import Workload
from raps.ui import LayoutManager
from raps.schedulers.rl import Scheduler
# from raps.resmgr.default import MultiTenantResourceManager as ResourceManager
from raps.resmgr.default import ExclusiveNodeResourceManager as ResourceManager


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

        resmgr = ResourceManager(
            total_nodes=self.config["TOTAL_NODES"],
            down_nodes=self.config.get("DOWN_NODES", []),
            config=self.config
        )

        # Plug in RL scheduler
        self.scheduler = Scheduler(
            config=self.config,
            policy="fcfs",   # or None if you want no heuristic fallback
            resource_manager=resmgr,
            env=self
        )
        self.engine.scheduler = self.scheduler

        self.layout_manager = LayoutManager(
            self.args_dict.get("layout"), engine=self.engine,
            debug=self.args_dict.get("debug", False),
            total_timesteps=self.args_dict.get("time", 1000),
            args_dict=self.args_dict,
            **self.config
        )

        self.generator = self.layout_manager.run_stepwise(
            jobs,
            timestep_start=0,
            timestep_end=self.config.get("SIM_END", 1000),
            time_delta=self.args_dict.get("time_delta", 1),
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

    def _compute_reward(self, tick_data, alpha=1.0, beta=0.001, gamma=0.1):
        completed = getattr(tick_data, "completed", None)
        jobs_completed = len(completed) if completed else 0
        power = self.power_manager.history[-1][1]
        queue_len = len(self.engine.queue)

        reward = alpha * jobs_completed - beta * power - gamma * queue_len

        if self.args_dict.get("debug", False):
            print(f"[t={self.engine.current_timestep}] jobs_completed={jobs_completed}, "
                  f"power={power}, queue_len={queue_len}, reward={reward}")

        return reward

    def step(self, action):
        chosen_job = None

        # Advance simulation by one step via generator
        try:
            tick_data = next(self.generator)
        except StopIteration:
            # Simulation finished
            return self._get_state(), 0.0, True, {}

        # Store action for scheduler to pick up
        self.scheduler.pending_action = action

        # Advance one step (scheduler.schedule() is called inside generator)
        tick_data = next(self.generator)
        reward = self._compute_reward(tick_data)

        obs = self._get_state()
        done = self.engine.current_timestep >= min(self.engine.timestep_end, 1000)

        info = {
            "scheduled_job": getattr(chosen_job, "id", None),
            "power": getattr(tick_data, "power", 0.0),
            "completed": getattr(tick_data, "completed", []),
        }
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
