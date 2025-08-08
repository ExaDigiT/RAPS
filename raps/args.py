import argparse
from raps.schedulers.default import PolicyType, BackfillType

from raps.workload import add_workload_to_parser, check_workload_args
from raps.utils import convert_to_seconds

parser = argparse.ArgumentParser(description='Resource Allocator & Power Simulator (RAPS)')

# System configurations
parser.add_argument('--system', type=str, default='frontier', help='System config to use')
parser.add_argument('-x', '--partitions', nargs='+', default=None, help='List of machine configurations to use, e.g., -x setonix-cpu setonix-gpu')
parser.add_argument('-c', '--cooling', action='store_true', help='Include FMU cooling model')
parser.add_argument('-net', '--simulate-network', default=False, action='store_true', help='Include Network model')

parser.add_argument('--noui', default=False, action='store_true', help='Run without UI')


# Simulation runtime options
parser.add_argument('-ff', '--fastforward', type=str, default=None, help='Fast-forward by time amount (uses same units as -t)')
parser.add_argument('-t', '--time', type=str, default=None, help='Length of time to simulate, e.g., 123, 123s, 27m, 3h, 7d')
#parser.add_argument("--time-delta", type=str, default=None, help='Time delta for simulation steps, e.g. 15, 15s 1m, 1h, 3d. (Default unit in seconds. If not set "TRACE_QUANTA" is used.)')  # This seems sensible, but 1s is the previous default before introducing this change!
parser.add_argument("--time-delta", type=str, default="1s", help='Time delta for simulation steps, e.g. 15, 15s 1m, 1h, 3d, 1ms. (Default unit in seconds. Default value: 1s.)')
parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode and disable rich layout')
parser.add_argument('-n', '--numjobs', type=int, default=100, help='Number of jobs to schedule')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--start', type=str, default='2021-05-21T13:00', help='ISO8061 string for start of simulation')
parser.add_argument('--end', type=str, default='2021-05-21T14:00', help='ISO8061 string for end of simulation')
parser.add_argument('--seed', action='store_true', help='Set random number seed for deterministic simulation')
parser.add_argument('-u', '--uncertainties', action='store_true',
                    help='Change from floating point units to floating point units with uncertainties.' + \
                         ' Very expensive w.r.t simulation time!')

# User Interface options
choices = ['layout1', 'layout2']
parser.add_argument('--layout', type=str, choices=choices, default=choices[0], help='Layout of UI')


# Output options
parser.add_argument('-o', '--output', action='store_true', help='Output power, cooling, and loss models for later analysis')
parser.add_argument('-p', '--plot', nargs='+', choices=['power', 'loss', 'pue', 'temp', 'util'],
                    help='Specify one or more types of plots to generate: power, loss, pue, util, temp')
choices = ['png', 'svg', 'jpg', 'pdf', 'eps']
parser.add_argument('--imtype', type=str, choices=choices, default=choices[0], help='Plot image type')

# Telemetry data
parser.add_argument('-f', '--replay', nargs='+', type=str, help='Either: path/to/joblive path/to/jobprofile' + \
                                                                ' -or- filename.npz (overrides --workload option)')
parser.add_argument('-e', '--encrypt', action='store_true', help='Encrypt any sensitive data in telemetry')
parser.add_argument('--validate', action='store_true', help='Use node power instead of CPU/GPU utilizations')
parser.add_argument('--jid', type=str, default='*', help='Replay job id')
parser.add_argument('--scale', type=int, default=0, help='Scale telemetry to max nodes specified in order to run telemetry on a smaller smaller target system/partition, e.g., --scale 192')

# Synthetic workloads
parser = add_workload_to_parser(parser)
#choices = ['random', 'benchmark', 'peak', 'idle','synthetic']
#parser.add_argument('-w', '--workload', type=str, choices=choices, default=choices[0], help='Type of synthetic workload')

# Scheduling options
choices = ['default', 'scheduleflow', 'nrel', 'anl', 'flux', 'experimental', 'multitenant']
parser.add_argument('--scheduler', type=str, choices=choices, default=choices[0], help='Name of scheduler')
choices = [policy.value for policy in PolicyType]
parser.add_argument('--policy', type=str, default=None, help='Schedule policy to use, e.g.:' + str(choices) + " or extended policies")
choices = [policy.value for policy in BackfillType]
parser.add_argument('--backfill', type=str, choices=choices, default=None, help='Backfill Policy')

# Redistribution of job arrival
choices = ['prescribed', 'poisson']
parser.add_argument('--arrival', default=choices[0], type=str, choices=choices, help=f'Modify arrival distribution ({choices[1]}) or use the original submit times ({choices[0]})')
parser.add_argument('--job-arrival-time', type=int, help='Modify job arrival for poisson distribution (in seconds). Overrides config/*/scheduler.json value.')  # no defaults as this overrides system config files
parser.add_argument('--job-arrival-rate', type=float, help='Modify arrival rate of poisson distribution (default 1)')  # no defaults as this overrides system config files


# Account options
parser.add_argument('--accounts', action='store_true', help='Flag indicating if accounts should be tracked')
parser.add_argument('--accounts-json', type=str, help='Json of account stats generated in previous run. see raps/accounts.py')


def post_process_args(args):
    if args.time_delta:
        time_delta_raw, time_delta_downscale_raw = convert_to_seconds(args.time_delta)
    else:
        time_delta_raw, time_delta_downscale_raw = None, 1

    if args.time:
        time_raw, time_downscale_raw = convert_to_seconds(args.time)
    else:
        time_raw, time_downscale_raw = None, 1

    if args.fastforward:
        ff_raw, ff_downscale_raw = convert_to_seconds(args.fastforward)
    else:
        ff_raw, ff_downscale_raw = None, 1

    max_downscale = max(time_delta_downscale_raw, time_downscale_raw, ff_downscale_raw)
    args.downscale = max_downscale

    if args.time_delta:
        args.time_delta = int((time_delta_raw / time_delta_downscale_raw) * max_downscale)
    if args.time:
        args.time = int((time_raw / time_downscale_raw) * max_downscale)
    if args.fastforward:
        args.fastforward = int((ff_raw / ff_downscale_raw) * max_downscale)

    return args


# ### At the end get args and an args_dict. import this if needed.
args = parser.parse_args()
# Do conversions and checks here if needed
check_workload_args(args)
args = post_process_args(args)
# generate the dictionary
args_dict = vars(args)
# #Now import args and args_dict directly if needed.:
# from args import args,args_dict
