def network_congestion(tx, rx, max_bw):
    """
    Overload factor ≥0: average of send/recv NOT clamped.
    >1.0 means you’re pushing above capacity.
    """
    tx_util = float(tx) / max_bw
    rx_util = float(rx) / max_bw
    return (tx_util + rx_util) / 2.0


def network_utilization(tx, rx, max_bw):
    """
    True utilization in [0,1]: average of send/recv clamped to 100%.
    """
    tx_u = min(float(tx) / max_bw, 1.0)
    rx_u = min(float(rx) / max_bw, 1.0)
    return (tx_u + rx_u) / 2.0


def network_slowdown(current_bw, max_bw):
    """
    Calculate a slowdown factor based on current network bandwidth usage.
    
    If current_bw is within limits, the factor is 1.0 (no slowdown).
    If current_bw exceeds max_bw, the factor is current_bw/max_bw.
    """
    if current_bw <= max_bw:
        return 1.0
    else:
        return current_bw / max_bw
