from raps.utils import convert_to_seconds
from raps.workload import Workload
from raps.scheduler import Scheduler
from raps.power import PowerManager, compute_node_power
from raps.flops import FLOPSManager
from raps.ui import LayoutManager
from raps.config import ConfigManager
import copy
from args import args
from raps.helpers import check_python_version
check_python_version()

args_dict1 = copy.deepcopy(vars(args))
args_dict2 = copy.deepcopy(vars(args))
print(args_dict1)
print(args_dict2)


config1 = ConfigManager(system_name='setonix-cpu').get_config()
config2 = ConfigManager(system_name='setonix-gpu').get_config()

args_dict1['config'] = config1
args_dict2['config'] = config2

pm1 = PowerManager(compute_node_power, **config1)
pm2 = PowerManager(compute_node_power, **config2)

fm1 = FLOPSManager(**args_dict1)
fm2 = FLOPSManager(**args_dict2)

sc1 = Scheduler(power_manager=pm1, flops_manager=fm1, cooling_model=None, **args_dict1)
sc2 = Scheduler(power_manager=pm2, flops_manager=fm2, cooling_model=None, **args_dict2)

layout_manager1 = LayoutManager(args.layout, scheduler=sc1, debug=args.debug, **config1)
layout_manager2 = LayoutManager(args.layout, scheduler=sc2, debug=args.debug, **config2)

print(config1)
print(config2)
configs = [config1, config2]
wl = Workload(*configs)

jobs = getattr(wl, args.workload)(num_jobs=args.numjobs)
print(jobs)

# Separate jobs based on partition
jobs1 = [job for job in jobs if job['partition'] == 'setonix-cpu']
jobs2 = [job for job in jobs if job['partition'] == 'setonix-gpu']

# Print counts for verification
print(f"Jobs for setonix-cpu: {len(jobs1)}")
print(f"Jobs for setonix-gpu: {len(jobs2)}")

if args.time:
    timesteps = convert_to_seconds(args.time)
else:
    timesteps = 88200  # 24 hours

if args.verbose:
    print(jobs)

# Create generator objects for both partitions
gen1 = layout_manager1.run_stepwise(jobs1, timesteps=timesteps)
gen2 = layout_manager2.run_stepwise(jobs2, timesteps=timesteps)

# Step through both generators in lockstep
# for _ in range(timesteps):
#    next(gen1)  # Advance first scheduler
#    next(gen2)  # Advance second scheduler

for timestep in range(timesteps):
    # Advance generators
    next(gen1)
    next(gen2)

    # Timestep
    print(f"[DEBUG] Timestep: {timestep}")

    # Queue lengths
    print(f"[DEBUG] setonix-cpu Queue: {len(layout_manager1.scheduler.queue)}")
    print(f"[DEBUG] setonix-gpu Queue: {len(layout_manager2.scheduler.queue)}")

    # System utilization
    sys_util1 = layout_manager1.scheduler.sys_util_history[-1][1] if layout_manager1.scheduler.sys_util_history else 0.0
    sys_util2 = layout_manager2.scheduler.sys_util_history[-1][1] if layout_manager2.scheduler.sys_util_history else 0.0
    print(f"[DEBUG] setonix-cpu Util: {sys_util1:.2f}%")
    print(f"[DEBUG] setonix-gpu Util: {sys_util2:.2f}%")
