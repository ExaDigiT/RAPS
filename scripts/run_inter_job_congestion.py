#!/usr/bin/env python3
"""
RAPS Network Congestion Test (Inter-Job Interference)
======================================================

This script is a wrapper that uses the integrated `inter_job_congestion`
workload from the RAPS library to run a standalone network simulation.

It evaluates inter-job network congestion by simulating multiple jobs
running concurrently on the same network and finding the total congestion
on the most loaded link.

Usage:
    python scripts/run_inter_job_congestion.py --config config/lassen.yaml

Example:
    python scripts/run_inter_job_congestion.py --config config/lassen.yaml --jobs 80 --txfrac 0.35 -v
"""

from __future__ import annotations
import argparse
from pathlib import Path

from raps.system_config import get_system_config
from raps.network import (
    NetworkModel,
    simulate_inter_job_congestion,
)
from raps.workloads import Workload


def print_verbose_stats(stats):
    print("\n--- Detailed Network Congestion Stats ---")
    print(f"  Max Congestion (Worst Link): {stats['max']:.2f}")
    print(f"  Mean Link Congestion:        {stats['mean']:.2f}")
    print(f"  Min Link Congestion:         {stats['min']:.2f}")
    print(f"  Std Dev of Congestion:       {stats['std_dev']:.2f}")
    print("\n  Top 10 Most Congested Links:")
    for (link, congestion) in stats['top_links']:
        print(f"    - Link {link}: {congestion:.2f}")
    print("---------------------------------------")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Standalone inter-job network congestion test for RAPS.")
    parser.add_argument("--config", required=True, help="Path to system YAML (e.g., config/lassen.yaml)")
    parser.add_argument("--jobs", type=int, default=60, help="Number of synthetic jobs")
    parser.add_argument("--txfrac", type=float, default=0.35, help="Fraction of per-link bandwidth per job")
    parser.add_argument("--debug", action="store_true", help="Enable network debug output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed statistics")
    args = parser.parse_args()

    # --- Load config and detect topology ---
    sys_cfg = get_system_config(args.config)
    legacy = sys_cfg.get_legacy()

    topology = legacy.get("TOPOLOGY", "").lower()
    if not topology:
        raise ValueError(f"Could not infer topology from {args.config}. Found: {topology!r}")

    # --- Generate Jobs via Workload module ---
    # The workload class expects specific attribute names, so we add them to the args object.
    args.workload = 'inter_job_congestion'
    args.numjobs = args.jobs
    args.seed = 42 # Keep seed consistent for this test script
    args.start = None

    workload_generator = Workload(args, legacy)
    workload_data = workload_generator.generate_jobs()
    jobs = workload_data.jobs

    print(f"[INFO] Detected topology: {topology}")
    print(f"[INFO] Generated {len(jobs)} jobs for congestion test.")

    # --- Initialize network model ---
    net = NetworkModel(
        config=legacy,
        available_nodes=list(range(legacy["TOTAL_NODES"])),
        output_dir=Path(f"test-{Path(args.config).stem}"),
        debug=args.debug,
    )
    
    # --- Simulate all jobs running concurrently ---
    congestion_stats = simulate_inter_job_congestion(net, jobs, legacy, debug=args.debug)

    print(f"[RESULT] config={args.config}, topology={topology}, jobs={len(jobs)}, "
          f"total_congestion={congestion_stats['max']:.2f}")

    if args.verbose:
        print_verbose_stats(congestion_stats)


if __name__ == "__main__":
    main()