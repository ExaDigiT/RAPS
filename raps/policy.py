from enum import Enum


class PolicyType(Enum):
    """Supported scheduling policies."""
    FCFS = 'fcfs'
    BACKFILL = 'backfill'
    PRIORITY = 'priority'
    FUGAKU_PTS = 'fugaku_pts'
    REPLAY = 'replay'
