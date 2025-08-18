#!/usr/bin/env python3
import os
import argparse
import json
import requests
import pandas as pd
from dotenv import load_dotenv

# ----------------------------------
# Environment / configuration
# ----------------------------------
load_dotenv()
# BASE_URL from env if you want, else localhost default
URL = os.getenv("BASE_URL", "http://localhost:8080")

# ----------------------------------
# Auth / HTTP helpers
# ----------------------------------
def read_token():
    token_path = ".api-token"
    if os.path.exists(token_path):
        try:
            with open(token_path, "r") as token_file:
                token = token_file.read().strip()
                if token:
                    return token
        except OSError as e:
            print(f"Warning: Could not read token file: {e}")
    # Fallback for localhost or dev use
    return "xyz123"


def call_api(endpoint, method="GET", params=None, data=None):
    token = read_token()
    url = f"{URL}{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = requests.request(method, url, headers=headers, params=params, json=data)
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None

    if resp.status_code == 200:
        # handle empty 200
        if not resp.content:
            return None
        try:
            return resp.json()
        except ValueError:
            print("Error: Response was 200 but not JSON")
            return None
    else:
        print(f"Error: {resp.status_code} - {resp.text}")
        return None

# ----------------------------------
# Command handlers
# ----------------------------------
def handle_run(args):
    # Build nested payload while omitting keys the user didn’t set
    data = {
        "start": args.start,
        "end": args.end,
        "system": args.system,
        "policy": args.policy,
        "parameters": args.parameters or {},
    }

    scheduler = {
        "enabled": args.scheduler_enabled,
        "num_jobs": args.scheduler_num_jobs,
        "seed": args.scheduler_seed,
        "jobs_mode": args.scheduler_jobs_mode,
    }
    scheduler = {k: v for k, v in scheduler.items() if v is not None}
    if scheduler:
        data["scheduler"] = scheduler

    cooling = {
        "enabled": args.cooling_enabled,
    }
    cooling = {k: v for k, v in cooling.items() if v is not None}
    if cooling:
        data["cooling"] = cooling

    response = call_api("/simulation/run", method="POST", data=data)
    print(response)

def handle_list(args):
    response = call_api("/simulation/list")
    if response:
        results = response.get("results", [])
        if not results:
            print("No simulations found.")
            return
        df = pd.DataFrame(results)
        # Feel free to uncomment for wider console displays:
        # pd.set_option('display.max_columns', None)
        # pd.set_option('display.max_colwidth', None)
        # pd.set_option('display.width', None)
        print(df)

def handle_simulation_details(args):
    response = call_api(f"/simulation/{args.id}")
    print(response)

def handle_cooling_cdu(args):
    response = call_api(f"/simulation/{args.id}/cooling/cdu")
    print(response)

def handle_cooling_cep(args):
    response = call_api(f"/simulation/{args.id}/cooling/cep")
    print(response)

def handle_scheduler_jobs(args):
    response = call_api(f"/simulation/{args.id}/scheduler/jobs")
    print(response)

def handle_power_history(args):
    response = call_api(f"/simulation/{args.id}/scheduler/jobs/{args.job_id}/power-history")
    print(response)

def handle_scheduler_system(args):
    response = call_api(f"/simulation/{args.id}/scheduler/system")
    print(response)

def handle_system_info(args):
    response = call_api(f"/system-info/{args.system}")
    print(response)

# ----------------------------------
# CLI
# ----------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Interact with the SimulationServer REST API.")
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Run simulation
    run_parser = subparsers.add_parser("run", help="Run a simulation.")

    # Top-level options
    run_parser.add_argument("--system", required=True, help="System to run the simulation on.")
    run_parser.add_argument("--policy", required=True, help="Policy to use.")
    run_parser.add_argument(
        "--parameters",
        type=json.loads,
        default={},
        help='Simulation parameters as JSON, e.g. \'{"alpha":0.1,"beta":"x"}\'',
    )
    run_parser.add_argument("--start", required=True, help="ISO time, e.g. 2024-01-01T00:00:00Z")
    run_parser.add_argument("--end", required=True, help="ISO time, e.g. 2024-01-01T00:10:00Z")

    # Scheduler group
    sched_grp = run_parser.add_argument_group("scheduler options")
    sched_grp.add_argument("--scheduler", dest="scheduler_enabled", action="store_true", help="Enable scheduler.")
    sched_grp.add_argument("--no-scheduler", dest="scheduler_enabled", action="store_false", help="Disable scheduler.")
    sched_grp.set_defaults(scheduler_enabled=None)  # omit if unspecified
    sched_grp.add_argument("--scheduler-num-jobs", type=int, help="Number of jobs.")
    sched_grp.add_argument("--scheduler-seed", type=int, help="Random seed.")
    sched_grp.add_argument("--scheduler-jobs-mode", choices=["random", "sequential"], help="Jobs mode.")

    # Cooling group
    cool_grp = run_parser.add_argument_group("cooling options")
    cool_grp.add_argument("--cooling", dest="cooling_enabled", action="store_true", help="Enable cooling.")
    cool_grp.add_argument("--no-cooling", dest="cooling_enabled", action="store_false", help="Disable cooling.")
    cool_grp.set_defaults(cooling_enabled=None)  # omit if unspecified

    run_parser.set_defaults(func=handle_run)

    # List simulations
    list_parser = subparsers.add_parser("list", help="List all simulations.")
    list_parser.set_defaults(func=handle_list)

    # Get simulation details
    details_parser = subparsers.add_parser("details", help="Get details of a simulation.")
    details_parser.add_argument("--id", required=True, help="Simulation ID.")
    details_parser.set_defaults(func=handle_simulation_details)

    # Cooling CDU
    cdu_parser = subparsers.add_parser("cooling-cdu", help="Get cooling CDU data for a simulation.")
    cdu_parser.add_argument("--id", required=True, help="Simulation ID.")
    cdu_parser.set_defaults(func=handle_cooling_cdu)

    # Cooling CEP
    cep_parser = subparsers.add_parser("cooling-cep", help="Get cooling CEP data for a simulation.")
    cep_parser.add_argument("--id", required=True, help="Simulation ID.")
    cep_parser.set_defaults(func=handle_cooling_cep)

    # Scheduler jobs
    jobs_parser = subparsers.add_parser("scheduler-jobs", help="Get scheduler jobs for a simulation.")
    jobs_parser.add_argument("--id", required=True, help="Simulation ID.")
    jobs_parser.set_defaults(func=handle_scheduler_jobs)

    # Power history
    power_parser = subparsers.add_parser("power-history", help="Get power history for a specific job in a simulation.")
    power_parser.add_argument("--id", required=True, help="Simulation ID.")
    power_parser.add_argument("--job-id", required=True, help="Job ID.")
    power_parser.set_defaults(func=handle_power_history)

    # Scheduler system
    scheduler_parser = subparsers.add_parser("scheduler-system", help="Get scheduler system data for a simulation.")
    scheduler_parser.add_argument("--id", required=True, help="Simulation ID.")
    scheduler_parser.set_defaults(func=handle_scheduler_system)

    # System info
    system_info_parser = subparsers.add_parser("system-info", help="Get system information.")
    system_info_parser.add_argument("--system", required=True, help="System name.")
    system_info_parser.set_defaults(func=handle_system_info)

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.command:
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
