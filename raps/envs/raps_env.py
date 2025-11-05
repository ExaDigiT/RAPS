import gym
from gym import spaces
import numpy as np

from raps.engine import Engine
from raps.stats import get_engine_stats, get_job_stats, get_scheduler_stats, get_network_stats

from stable_baselines3.common.logger import Logger, HumanOutputFormat
import sys

logger = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])


def print_stats(stats, step=0):
    """prints SB3-style stats output"""

    wanted_keys = {
        "time_simulated": "engine/Time Simulated",
        "average_power": "engine/Average Power",
        "system_power_efficiency": "engine/System Power Efficiency",
        "total_energy_consumed": "engine/Total Energy Consumed",
        "carbon_emissions": "engine/Carbon Footprint",
        "jobs_completed": "jobs/Jobs Completed",
        "throughput": "jobs/Throughput",
        "jobs_still_running": "jobs/Jobs Still Running",
    }

    for section in ["engine_stats", "job_stats"]:
        if section in stats:
            for k, v in stats[section].items():
                if k in wanted_keys:
                    if k == "jobs_still_running" and isinstance(v, list):
                        v = len(v)
                    logger.record(wanted_keys[k], v)

    logger.dump(step=step)


class RAPSEnv(gym.Env):
    """
    Minimal Gym-compatible wrapper around RAPS Engine
    for RL job scheduling experiments.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, sim_config):
        super().__init__()
        # Store everything in self.args
        self.sim_config = sim_config
        self.engine = self._create_engine()

        # --- RL spaces ---
        max_jobs = 100
        job_features = 4  # [nodes, runtime, priority, wait_time]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(max_jobs, job_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(max_jobs)

    def _create_engine(self):
        engine = Engine(self.sim_config)
        engine.scheduler.env = self
        self.jobs = engine.jobs
        self.generator = engine.run_simulation()
        return engine

    def reset(self, **kwargs):
        self.engine = self._create_engine()
        obs = self._get_state()
        return obs

    def _compute_reward(self, tick_data):
        """
        Reward function for RL scheduling on Frontier-like systems.
        Balances throughput and carbon footprint, using incremental values.
        """

        # How many jobs completed *this tick*
        jobs_done = len(getattr(tick_data, "completed", []))

        # Incremental carbon emitted this tick
        carbon_step = getattr(self.engine, "carbon emissions", 0.0)

        # Tradeoff weights (tunable hyperparameters)
        alpha = 10.0   # reward for finishing a job
        beta = 0.1    # penalty per metric ton CO2

        # Reward = (jobs * alpha) - (carbon * beta)
        reward = (alpha * jobs_done) - (beta * carbon_step)

        # Small penalty if idle and no jobs complete
        if jobs_done == 0 and carbon_step == 0:
            reward -= 0.01

        return reward

    def step(self, action):
        if self.engine is None:
            raise RuntimeError("Engine not initialized. Did you forget to call reset()?")

        queue = self.engine.queue
        invalid_action = False

        # If queue empty or index out of range → invalid
        if len(queue) == 0 or action >= len(queue):
            invalid_action = True
        else:
            job = queue[int(action)]
            available_nodes = self.engine.scheduler.resource_manager.available_nodes

            if job.nodes_required <= len(available_nodes):
                # Just pick the first available node (simplest placement policy)
                node_id = available_nodes[0]
                self.engine.scheduler.place_job_and_manage_queues(
                    job,
                    queue,
                    self.engine.running,
                    self.engine.current_timestep,
                    node_id,
                )
            else:
                invalid_action = True

        # advance simulation by one tick
        tick_data = next(self.generator)

        # compute reward
        if invalid_action:
            reward = -1.0
        else:
            reward = self._compute_reward(tick_data)

        # clip reward
        reward = np.clip(reward, -10.0, 10.0)

        # Print stats
        stats = self.get_stats()
        print_stats(stats)

        obs = self._get_state()
        done = self.engine.current_timestep >= self.engine.timestep_end
        info = {}

        print(f"t={self.engine.current_timestep}, "
              f"queue={len(self.engine.queue)}, "
              f"running={len(self.engine.running)}, "
              f"completed={self.engine.jobs_completed}",
              f"action={action}")

        return obs, reward, done, info

    def _get_state(self):
        """Construct simple state representation from engine's job queue."""
        # Example: take waiting jobs (haven’t started yet)
        job_queue = [j for j in self.jobs if getattr(j, "start_time", None) is None]

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

    def get_stats(self):
        return {
            "engine_stats": get_engine_stats(self.engine),
            "job_stats": get_job_stats(self.engine),
            "scheduler_stats": get_scheduler_stats(self.engine),
            "network_stats": get_network_stats(self.engine)
        }
