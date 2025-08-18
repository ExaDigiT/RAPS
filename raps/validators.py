def recompute_power(nodes, running_jobs, current_time):
    node_power = {n['id']: 0.0 for n in nodes}
    for j in running_jobs:
        idx = max(0, current_time - j.start_time)
        # Clamp index
        idx = min(idx, len(j.cpu_trace)-1)
        cpu_p = j.cpu_trace[idx]
        gpu_p = j.gpu_trace[idx] if j.gpu_trace else 0
        nid = j.scheduled_nodes[0]
        node_power[nid] += cpu_p + gpu_p
    total = sum(node_power.values())
    return node_power, total
