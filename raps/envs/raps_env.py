import copy
import gym
from gym import spaces
import numpy as np

from raps.engine import Engine
from raps.workload import Workload
# from raps.resmgr.default import MultiTenantResourceManager as ResourceManager
from raps.stats import get_engine_stats, get_job_stats, get_scheduler_stats, get_network_stats

from stable_baselines3.common.logger import Logger, HumanOutputFormat
import sys

logger = Logger(folder=None, output_formats=[HumanOutputFormat(sys.stdout)])


def print_stats(stats, step=0):
    """prints SB3-style stats output"""

    wanted_keys = {
        "time simulated": "engine/Time Simulated",
        "average power": "engine/Average Power",
        "system power efficiency": "engine/System Power Efficiency",
        "total energy consumed": "engine/Total Energy Consumed",
        "carbon emissions": "engine/Carbon Footprint",
        "jobs completed": "jobs/Jobs Completed",
        "throughput": "jobs/Throughput",
        "jobs still running": "jobs/Jobs Still Running",
    }

    for section in ["engine_stats", "job_stats"]:
        if section in stats:
            for k, v in stats[section].items():
                if k.lower() in wanted_keys:
                    if k.lower() == "jobs still running" and isinstance(v, list):
                        v = len(v)
                    logger.record(wanted_keys[k.lower()], v)

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
        self.engine, workload_data, time_delta = Engine.from_sim_config(self.sim_config)
        self.engine.scheduler.env = self
        jobs = workload_data.jobs
        timestep_start = workload_data.telemetry_start
        timestep_end = workload_data.telemetry_end
        self.generator = self.engine.run_simulation(jobs, timestep_start, timestep_end, time_delta)

    def _build_jobs(self):
        """
        Build a job list either from synthetic workload (--workload)
        or from telemetry replay (--replay).
        Returns: jobs, timestep_start, timestep_end
        """
        # --- Case 1: Telemetry replay ---
        if self.cli_args and getattr(self.cli_args, "replay"):
            result = self.telemetry.load_jobs_times_args_from_files(
                files=self.cli_args.replay,
                args=self.cli_args,
                config=self.config,
            )

            # Handle 3-tuple vs 4-tuple return
            if len(result) == 3:
                jobs, start_time, end_time = result
            elif len(result) == 4:
                jobs, start_time, end_time, _ = result
            else:
                raise ValueError(f"Unexpected telemetry return format: {len(result)} values")

            # Flatten partitioned jobs if necessary
            if jobs and isinstance(jobs[0], list):
                jobs = [job for sublist in jobs for job in sublist]

            return jobs, start_time, end_time

        # --- Case 2: Synthetic workload generation ---
        elif self.cli_args and getattr(self.cli_args, "workload"):
            wl = Workload(self.cli_args, self.config)
            jobs = wl.generate_jobs()

            # For synthetic jobs, compute timestep_end from submit + run_time
            timestep_start = 0
            timestep_end = max(
                (getattr(job, "end_time", None) or getattr(job, "expected_run_time", 0) + job.submit_time)
                for job in jobs
            )
            return jobs, timestep_start, timestep_end

        # --- Error: neither replay nor workload specified ---
        else:
            raise ValueError("RAPSEnv requires either --workload or --replay to build jobs.")

#    def reset(self, seed=None, options=None):
#        super().reset(seed=seed)
#
#        self.jobs = copy.deepcopy(self.original_jobs)  # working copy
#
#        # Reset engine
#        self.engine.current_timestep = 0
#        #self.engine.reset()  # or clear state manually
#        power_manager = PowerManager(compute_node_power, **self.config)
#        flops_manager = FLOPSManager(**self.args_dict)
#        telemetry = Telemetry(**self.args_dict)
#        jobs, timestep_start, timestep_end = self._build_jobs()
#
#        self.engine = Engine(
#            power_manager=power_manager,
#            flops_manager=flops_manager,
#            jobs=jobs,
#            **self.args_dict
#        )
#
#        self.engine.timestep_start = timestep_start
#        self.engine.timestep_end = timestep_end
#        #self.engine.current_timestep = timestep_start
#
#        # Restart generator
#        self.generator = self.layout_manager.run_stepwise(
#            self.jobs,
#            timestep_start=self.timestep_start,
#            timestep_end=self.timestep_end,
#            time_delta=self.args_dict.get("time_delta"),
#        )
#
#        return self._get_state(), {}

    def reset(self, **kwargs):
        self.engine = self._create_engine()

    def reset2(self, **kwargs):
        completed = [j.id for j in self.jobs if j.current_state.name == "COMPLETED"]
        print(f"[RESET] Jobs already completed before deepcopy: {len(completed)}")

        super().reset(seed=42)
        # self.engine.jobs = self.jobs
        self.jobs = copy.deepcopy(self.original_jobs)  # working copy

        # self.engine.timestep_start = self.timestep_start
        # self.engine.timestep_end = self.timestep_end
        # self.engine.reset(self.jobs, self.timestep_start, self.timestep_end)

        # self.engine.current_timestep = self.timestep_start

        # self.engine.jobs = self.jobs  # repoint engine to fresh jobs
        # self.engine.completed_jobs = []
        # self.engine.queue.clear()
        # self.engine.running.clear()
        # self.engine.power_manager.history.clear()
        # self.engine.jobs_completed = 0

        self.generator = self.layout_manager.run_stepwise(
            self.jobs,
            timestep_start=self.timestep_start,
            timestep_end=self.timestep_end,
            time_delta=self.args_dict.get("time_delta", 1),
        )

        return self._get_state()

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

#    def _compute_reward(self, tick_data):
#        """
#        Reward function: minimize carbon footprint per job completed.
#        Encourages the agent to complete jobs while keeping emissions low.
#        """
#        reward = 0.0
#
#        # Jobs completed this tick
#        jobs_completed = len(getattr(tick_data, "completed", []))
#
#        # Carbon emitted so far (metric tons CO2)
#        carbon_so_far = getattr(self.engine, "carbon emissions", 0.0)
#
#        if jobs_completed > 0:
#            # Reward is higher when more jobs finish with less carbon
#            reward = jobs_completed / (carbon_so_far + 1e-6)
#        else:
#            # Small penalty if no jobs finished (encourages progress)
#            reward = -0.01
#
#        return reward

    def _compute_reward2(self, tick_data, alpha=10.0, beta=1.0, gamma=2.0):
        completed = getattr(tick_data, "completed", None)
        jobs_completed = len(completed) if completed else 0
        power = self.power_manager.history[-1][1]
        queue_len = len(self.engine.queue)

        reward = alpha * jobs_completed - beta * power - gamma * queue_len

        print(f"[t={self.engine.current_timestep}] jobs_completed={jobs_completed}, "
              f"power={power}, queue_len={queue_len}, reward={reward}")

        return reward

    def step(self, action):
        queue = self.engine.queue
        invalid_action = False

        # If queue empty or index out of range → invalid
        if len(queue) == 0 or action >= len(queue):
            invalid_action = True
        else:
            job = queue[int(action)]
            available = len(self.engine.scheduler.resource_manager.available_nodes)
            if job.nodes_required <= available:
                # Valid scheduling
                self.engine.scheduler.place_job_and_manage_queues(
                    job, queue, self.engine.running, self.engine.current_timestep
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
