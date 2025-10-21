"""
See raps/dataloaders/philly.py for how to download philly traces.

Run following to parse philly traces into separate files for each day:

    python /path/to/raps/scripts/parse_philly_traces.py cluster_cpu_util
    python /path/to/raps/scripts/parse_philly_traces.py cluster_gpu_util

This will parse these two files into two directories, cpu_by_day and gpu_by_day,
creating one file for each day and adding the lines for that day into the files.
"""
import os
import sys
from datetime import datetime
from tqdm import tqdm

if len(sys.argv) < 2:
    print("Usage: python parse_by_day.py <input_file>")
    sys.exit(1)

input_file = sys.argv[1]

with open(input_file) as f:
    total_lines = sum(1 for _ in f) - 1

with open(input_file) as f:
    header = f.readline().strip().split(",")
    print("Header:", header)

    # detect file type from header
    is_cpu = "cpu_util" in [h.lower() for h in header]

    # pick output dir name based on file type
    output_dir = "cpu_by_day" if is_cpu else "gpu_by_day"
    os.makedirs(output_dir, exist_ok=True)

    for line in tqdm(f, total=total_lines, desc="Processing lines"):
        parts = line.strip().split(",")

        if len(parts) < 3:
            continue

        raw_time = parts[0].replace(" PST", "").replace(" PDT", "")
        try:
            ts = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

        machine_id = parts[1]

        if is_cpu:
            try:
                value = float(parts[2])
            except ValueError:
                value = 0.0
            label = "cpu_util"
        else:
            utils = []
            for v in parts[2:]:
                try:
                    utils.append(float(v))
                except ValueError:
                    pass
            value = sum(utils) / max(1, len([u for u in utils if u > 0]))
            label = "gpu_util"

        day_str = ts.strftime("%Y-%m-%d")
        out_path = os.path.join(output_dir, f"{day_str}.csv")

        with open(out_path, "a") as out:
            if out.tell() == 0:  # only write header if file is new
                out.write(f"time,machine_id,{label}\n")
            out.write(f"{ts},{machine_id},{value:.3f}\n")
