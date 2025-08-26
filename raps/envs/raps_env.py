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

    def _compute_reward(self, tick_data, alpha=1.0, beta=0.001, gamma=0.1):
        completed = getattr(tick_data, "completed", None)
        jobs_completed = len(completed) if completed else 0
        power = getattr(tick_data, "power", 0.0) or 0.0
        queue_len = len(self.engine.queue)

        reward = alpha * jobs_completed - beta * power - gamma * queue_len

        if self.args_dict.get("debug", False):
            print(f"[t={self.engine.current_timestep}] jobs_completed={jobs_completed}, "
                  f"power={power}, queue_len={queue_len}, reward={reward}")

        return reward

    def step(self, action):
        # 1. Jobs waiting in the queue
        job_queue = list(self.engine.queue)
        chosen_job = None

        if job_queue and action < len(job_queue):
            chosen_job = job_queue[action]

            # 2. Let RAPS handle all scheduling logic
            self.engine.scheduler.place_job_and_manage_queues(
                chosen_job,
                self.engine.queue,
                self.engine.running,
                self.engine.current_timestep,
            )

        # 3. Advance simulation by one tick
        # Update bookkeeping so tick() doesn't crash
        if not hasattr(self.engine, "num_active_nodes"):
            self.engine.num_active_nodes = 0
        if not hasattr(self.engine, "num_free_nodes"):
            self.engine.num_free_nodes = self.config["AVAILABLE_NODES"]

        self.engine.num_active_nodes = sum(len(j.scheduled_nodes) for j in self.engine.running)
        self.engine.num_free_nodes = self.config["AVAILABLE_NODES"] - self.engine.num_active_nodes

        tick_data = self.engine.tick()

        # 4. Compute reward (throughput vs. power)
        reward = self._compute_reward(tick_data)

        # 5. Build next observation
        obs = self._get_state()
        done = self.engine.current_timestep >= self.engine.timestep_end

        info = {
            "scheduled_job": getattr(chosen_job, "id", None),
            "power": getattr(tick_data, "power", None),
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
