def network_utilization(tx, rx, MAX):
    """Compute average network utilization"""
    tx_util = float(tx) / MAX
    rx_util = float(rx) / MAX
    return (tx_util + rx_util) / 2.0

def network_dilation_factor(current_bw, max_bw):
    """
    Calculate a dilation factor based on current network bandwidth usage.
    
    If current_bw is within limits, the factor is 1.0 (no slowdown).
    If current_bw exceeds max_bw, the factor is current_bw/max_bw.
    """
    if current_bw <= max_bw:
        return 1.0
    else:
        return current_bw / max_bw

def get_current_bandwidth_usage(link_id):
    """
    Placeholder function: In a real system, query the current bandwidth usage
    for the given network link. Here we return a fixed value for demonstration.
    """
    return 150.0  # e.g., 150
