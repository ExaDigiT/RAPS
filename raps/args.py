import argparse, os, sys, yaml
from raps.schedulers.default import PolicyType, BackfillType

from raps.workload import add_workload_to_parser, check_workload_args
from raps.utils import convert_to_seconds


def load_config(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def _expand_path(p):
    if isinstance(p, str):
        # expand ~ and $VARS
        return os.path.expanduser(os.path.expandvars(p))
    return p


def apply_config_to_args(cfg, args):
    # Merge supported sections or top-level keys
    merged = {}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and k in {
            "shared", "simulate", "telemetry", "scheduler", "output"
        }:
            merged.update(v)
        else:
            # Enter the commandline argument, but _underscores as the -dashes
            # are replaced when reading from the commandline, but not in the yaml.
            merged[k.replace('-','_')] = v

    # Apply to argparse namespace
    for k, v in merged.items():
        setattr(args, k, v)

    # Coerce certain keys to lists if YAML provided strings
    list_keys = {
        "cluster_var", "output_vars", "input_vars", "partitions", "plot"
    }
    for key in list_keys:
        if hasattr(args, key):
            val = getattr(args, key)
            if isinstance(val, str):
                setattr(args, key, [val])

    # Expand paths (tilde + env vars)
    for key in ("path", "output_dir", "plot_dir", "config_file"):
        if hasattr(args, key):
            setattr(args, key, _expand_path(getattr(args, key)))

    # Normalize enums if provided as strings in YAML
    if getattr(args, "policy", None):
        try:
            # Accept exact values or case-insensitive
            val = str(args.policy)
            opts = {p.value.lower(): p.value for p in PolicyType}
            if val.lower() in opts:
                args.policy = opts[val.lower()]
        except Exception:
            pass

    if getattr(args, "backfill", None):
        try:
            val = str(args.backfill)
            opts = {b.value.lower(): b.value for b in BackfillType}
            if val.lower() in opts:
                args.backfill = opts[val.lower()]
        except Exception:
            pass


parser = argparse.ArgumentParser(
    description="Resource Allocator & Power Simulator (RAPS)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "config_file", nargs="?", default=None,
    help="YAML config file; overrides defaults/flags."
)

# System configurations
parser.add_argument("--system", type=str, default="frontier",
                    help="System config to use")
parser.add_argument(
    "-x", "--partitions", nargs="+", default=None,
    help="List of machine configurations, e.g., -x setonix-cpu setonix-gpu"
)
parser.add_argument("-c", "--cooling", action="store_true",
                    help="Include FMU cooling model")
parser.add_argument("-net", "--simulate-network", default=False,
                    action="store_true", help="Include Network model")
parser.add_argument("--noui", default=False, action="store_true",
                    help="Run without UI")

# Simulation runtime options
parser.add_argument("-ff", "--fastforward", type=str, default=None,
                    help="Fast-forward by time amount (uses same units as -t)")
parser.add_argument("-t", "--time", type=str, default=None,
                    help="Length of time to simulate, e.g., 123, 27m, 3h, 7d")
parser.add_argument("--time-delta", type=str, default="1s",
                    help="Step size, e.g., 15s, 1m, 1h, 1ms (default: 1s)")
parser.add_argument("-d", "--debug", action="store_true",
                    help="Enable debug mode and disable rich layout")
parser.add_argument("-n", "--numjobs", type=int, default=100,
                    help="Number of jobs to schedule")
parser.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose output")
parser.add_argument("--start", type=str, default="2021-05-21T13:00",
                    help="ISO8601 start of simulation")
parser.add_argument("--end", type=str, default="2021-05-21T14:00",
                    help="ISO8601 end of simulation")
parser.add_argument("--seed", action="store_true",
                    help="Set RNG seed for deterministic simulation")
parser.add_argument(
    "-u", "--uncertainties", action="store_true",
    help=("Use float-with-uncertainties (much slower).")
)

# UI
ui_layout_choices = ["layout1", "layout2"]
parser.add_argument("--layout", type=str, choices=ui_layout_choices,
                    default=ui_layout_choices[0], help="UI layout")

# Output
parser.add_argument('-o', '--output', type=str, nargs="?",
                    const="",  # Used if -o is given without a value
                    default=None,     # Used if -o is not provided at all
                    help=("Output power, cooling, and loss models for later ",
                          "analysis. Argumment specifies name."),
                    )
plot_choices = ["power", "loss", "pue", "temp", "util"]
parser.add_argument("-p", "--plot", nargs="+", choices=plot_choices,
                    help="Plots to generate")
img_choices = ["png", "svg", "jpg", "pdf", "eps"]
parser.add_argument("--imtype", type=str, choices=img_choices,
                    default=img_choices[0], help="Plot image type")

# Telemetry
parser.add_argument(
    "-f", "--replay", nargs="+", type=str,
    help=("Either: path/to/joblive path/to/jobprofile OR filename.npz "
          "(overrides --workload)")
)
parser.add_argument("-e", "--encrypt", action="store_true",
                    help="Encrypt sensitive data in telemetry")
parser.add_argument("--validate", action="store_true",
                    help="Use node power instead of CPU/GPU utilizations")
parser.add_argument("--jid", type=str, default="*",
                    help="Replay job id")
parser.add_argument("--scale", type=int, default=0,
                    help=("Scale telemetry to a smaller target system, "
                          "e.g., --scale 192"))

# Synthetic workloads
parser = add_workload_to_parser(parser)

# Scheduling
sched_choices = ["default", "scheduleflow", "nrel", "anl", "flux",
                 "experimental", "multitenant"]
parser.add_argument("--scheduler", type=str, choices=sched_choices,
                    default=sched_choices[0], help="Scheduler name")
parser.add_argument("--policy", type=str, default=None,
                    help=f"Schedule policy: {[p.value for p in PolicyType]}")
parser.add_argument("--backfill", type=str, default=None,
                    help=f"Backfill policy: {[b.value for b in BackfillType]}")

# Arrival
arr_choices = ["prescribed", "poisson"]
parser.add_argument("--arrival", default=arr_choices[0], type=str,
                    choices=arr_choices,
                    help=("Modify arrival distribution (poisson) or use "
                          "original submit times (prescribed)"))
parser.add_argument("--job-arrival-time", type=int,
                    help=("Poisson arrival (seconds). Overrides "
                          "config/*/scheduler.json"))
parser.add_argument("--job-arrival-rate", type=float,
                    help="Modify Poisson rate (default 1)")

# Accounts
parser.add_argument("--accounts", action="store_true",
                    help="Track accounts")
parser.add_argument("--accounts-json", type=str,
                    help="Accounts JSON from previous run")

# Downtime
parser.add_argument("--downtime-first", type=str, default=None,
                    help="First downtime, e.g., after 123, 27m, 3h, 7d")
parser.add_argument("--downtime-interval", type=str, default=None,
                    help="Interval between downtimes, e.g., every 123, 27m, 3h, 7d")
parser.add_argument("--downtime-length", type=str, default=None,
                    help="Downtime length, e.g., 123, 27m, 3h, 7d")

# Continous Job Generation
parser.add_argument("--continuous-job-generation", action="store_true",
                    help="Activate continuous job generation.")
parser.add_argument("--maxqueue", type=int, default=50,
                    help="Specify the max queue length for continuous job generation.")



def post_process_args(args):
    if args.time_delta:
        tdelta_raw, tdelta_down = convert_to_seconds(args.time_delta)
    else:
        tdelta_raw, tdelta_down = None, 1

    if args.time:
        time_raw, time_down = convert_to_seconds(args.time)
    else:
        time_raw, time_down = None, 1

    if args.fastforward:
        ff_raw, ff_down = convert_to_seconds(args.fastforward)
    else:
        ff_raw, ff_down = None, 1

    if args.downtime_first:
        dtf_raw, dtf_down = convert_to_seconds(args.downtime_first)
    if args.downtime_interval:
        dti_raw, dti_down = convert_to_seconds(args.downtime_interval)
    if args.downtime_length:
        dtl_raw, dtl_down = convert_to_seconds(args.downtime_length)

    max_down = max(tdelta_down, time_down, ff_down)
    args.downscale = max_down

    if args.time_delta:
        args.time_delta = int((tdelta_raw / tdelta_down) * max_down)
    if args.time:
        args.time = int((time_raw / time_down) * max_down)
    if args.fastforward:
        args.fastforward = int((ff_raw / ff_down) * max_down)

    if args.downtime_first:
        args.downtime_first = int((dtf_raw / dtf_down) * max_down)
    if args.downtime_interval:
        args.downtime_interval = int((dti_raw / dti_down) * max_down)
    if args.downtime_length:
        args.downtime_length = int((dtl_raw / dtl_down) * max_down)

    return args


# ---- Parse + YAML merge ----
args = parser.parse_args()

# Config file existence check
if args.config_file and not os.path.isfile(args.config_file):
    print(f"Error: '{args.config_file}' not found.", file=sys.stderr)
    sys.exit(1)

cfg = load_config(args.config_file)

apply_config_to_args(cfg, args)

# Optional: format fileprefix after config merge (if provided by workload parser)
if hasattr(args, "fileprefix") and isinstance(args.fileprefix, str):
    try:
        args.fileprefix = args.fileprefix.format(**vars(args))
    except KeyError as e:
        print(f"Warning: missing placeholder {e} in fileprefix; skipping.")

# Expand paths inside list fields (e.g., replay)
if getattr(args, "replay", None):
    if isinstance(args.replay, str):
        args.replay = [args.replay]
    args.replay = [_expand_path(p) for p in args.replay]

# Prefer replay if both replay and workload got set
if getattr(args, "replay", None) and getattr(args, "workload", None):
    print("Info: --replay provided; ignoring --workload.", file=sys.stderr)
    print("Info: --replay provided; ignoring --workload.", file=sys.stderr)
    args.workload = None

# Enforce valid policy/backfill values (after normalization in apply_config_to_args)
if getattr(args, "policy", None):
    _valid_policies = {p.value for p in PolicyType}
    if args.policy not in _valid_policies:
        sys.exit(f"Error: Unknown policy '{args.policy}'. "
                 f"Valid: {sorted(_valid_policies)}")
if getattr(args, "backfill", None):
    _valid_backfills = {b.value for b in BackfillType}
    if args.backfill not in _valid_backfills:
        sys.exit(f"Error: Unknown backfill '{args.backfill}'. "
                 f"Valid: {sorted(_valid_backfills)}")

# Multi-partition guard for single-part driver (check merged args incl. CLI)
if os.path.basename(sys.argv[0]) == "main.py":
    _parts = args.partitions or []
    if isinstance(_parts, str):
        _parts = [_parts]
    if len(_parts) > 1:
        sys.exit("Error: Use multi-part-sim.py for multi-partition runs.")

# Validate workload args before time conversions
check_workload_args(args)

# Convert time-like args and compute downscale
args = post_process_args(args)

# Expose dict form
args_dict = vars(args)
