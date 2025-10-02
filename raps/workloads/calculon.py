"""
Calculon is a analytical model for estimating LLM training times for given architectures
on particular hardware. It is described in the paper:

    Isaev, Mikhail, et al. "Calculon: a methodology and tool for high-level co-design of 
    systems and large language models." SC23 Proceedings
    https://dl.acm.org/doi/pdf/10.1145/3581784.3607102

The code is available at https://github.com/calculon-ai/calculon
which this module assumes is already cloned into the third_party directory.

Calculon requires installing `psutil`, which can be pip installed via:

    pip install psutil

Since Calculon by default supports A100 GPUs, we are able to use the default files that
are already setup in Calculon, and therefore have added two systems which have A100 GPUs:
Selene and Perlmutter. Example run commands:

    python main.py run --system selene -w calculon
    python main.py run --system perlmutter -w calculon

This code is currently setup to generate synthetic traces for four different LLM models:
megatron-22B, gpt3-175B, turing-530B, and megatron-1T. Adjust these by modifying 
llm_model_tests below.

"""
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np

from raps.job import job_dict

from .constants import ACCT_NAMES


class Calculon:
    """Calculon workload mixin for Workload class."""

    def __init__(self, *args, **kwargs):
        # NOTE: mixins usually accept (sim_config_args, system_config_dict) through Workload
        super().__init__(*args, **kwargs)

    def calculon(self, **kwargs):
        """Generate workload using Calculon backend + job trace synthesis."""
        jobs = []

        llm_models_test = [
            ["megatron-22B", 8, 4],
            ["gpt3-175B", 64, 64],
            ["turing-530B", 280, 280],
            ["megatron-1T", 512, 512],
        ]

        for llm_model, num_nodes, max_batch_size in llm_models_test:
            for partition in self.partitions:
                config = self.config_map[partition]
                gpu_system = "a100_80g"
                data_type = "float16"
                output = f"{llm_model}_{gpu_system}_{max_batch_size}_{data_type}_{num_nodes}.json"

                # call Calculon binary/subprocess to get MFU + batch time
                mfu, total_batch_time = self._run_calculon(
                    llm_model, gpu_system, max_batch_size, num_nodes, data_type, output
                )

                # derive job stats
                num_iters = 3000
                trace_quanta = config["TRACE_QUANTA"]
                job_time = total_batch_time * num_iters
                num_samples = int(job_time // trace_quanta)

                system_util = np.full(num_samples, mfu)
                cpu_util = random.random() * config["CPUS_PER_NODE"]
                cpu_trace = cpu_util * np.ones(num_iters)

                net_tx, net_rx = [], []
                num_nodes = num_nodes // config["GPUS_PER_NODE"]

                epochs = 1
                wall_time = job_time
                for i in range(epochs):
                    job_info = job_dict(
                        nodes_required=num_nodes,
                        scheduled_nodes=[],
                        name=f"{llm_model} training for {num_iters} iterations",
                        account=ACCT_NAMES[0],
                        cpu_trace=cpu_trace,
                        gpu_trace=system_util,
                        ntx_trace=net_tx,
                        nrx_trace=net_rx,
                        end_state="COMPLETED",
                        id=None,
                        priority=100,
                        partition=partition,
                        time_limit=job_time + 1,
                        start_time=0,
                        end_time=job_time,
                        trace_time=job_time,
                        trace_start_time=0,
                        trace_end_time=job_time,
                    )
                    jobs.append(job_info)
                    wall_time += job_time

        return jobs

    def _run_calculon(self, model, system, max_batch_size, num_nodes, data_type, output):
        """Internal: run Calculon subprocess and parse result."""
        base_path = Path("third_party/calculon")

        # paths
        model_path = base_path / "models" / f"{model}.json"
        system_path = base_path / "systems" / f"{system}.json"
        raw_path   = base_path / "optimal_executions" / output.replace(".json", "_raw.json")
        exec_path  = base_path / "optimal_executions" / output.replace(".json", "_exec.json")
        stats_path = base_path / "optimal_executions" / output.replace(".json", "_stats.json")

        # Run llm-optimal-execution to generate candidate executions
        opt_cmd = [
            "./bin/calculon", "llm-optimal-execution",
            f"models/{model}.json",
            str(num_nodes),
            str(max_batch_size),
            data_type,
            f"systems/{system}.json",
            f"optimal_executions/{output.replace('.json', '_raw.json')}",
        ]
        subprocess.run(opt_cmd, check=True, cwd=base_path, env={**os.environ, "PYTHONPATH": "."})

        # Read raw output, pick first/best execution and dump it as exec.json
        with open(raw_path) as f:
            raw_data = json.load(f)

        # get first (or best) key
        first_key = sorted(raw_data.keys(), key=lambda k: float(k))[0]
        best_exec = raw_data[first_key]["execution"]

        with open(exec_path, "w") as f:
            json.dump(best_exec, f, indent=2)

        # Run llm with chosen execution, system, and model → stats.json
        llm_cmd = [
            "./bin/calculon", "llm",
            f"models/{model}.json",
            f"optimal_executions/{output.replace('.json', '_exec.json')}",
            f"systems/{system}.json",
            f"optimal_executions/{output.replace('.json', '_stats.json')}",
        ]
        subprocess.run(llm_cmd, check=True, cwd=base_path, env={**os.environ, "PYTHONPATH": "."})

        # Parse stats.json to extract metrics
        with open(stats_path) as f:
            stats_data = json.load(f)

        stats = stats_data.get("stats", {})

        # These keys may vary depending on Calculon version
        mfu = stats.get("model_flops_utilization") \
            or stats.get("sample_rate") \
            or stats.get("best_sample_rate") \
            or 0.0

        total_batch_time = stats.get("block_fw_time") \
            or stats.get("batch_time") \
            or stats.get("total_time") \
            or 0.0

        return mfu, total_batch_time
