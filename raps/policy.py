from enum import Enum


class PolicyType(Enum):
    """Supported scheduling policies."""
    FCFS = 'fcfs'
    PRIORITY = 'priority'
    FUGAKU_PTS = 'fugaku_pts'
    REPLAY = 'replay'
    SJF = 'sjf'
    LJF = 'ljf'

class BackfillType(Enum):
    """Supported backfilling policies."""
    NONE = None
    FIRSTFIT = 'firstfit'
    BESTFIT = 'bestfit'
    GREEDY = 'greedy'
    EASY = 'easy'  # Earliest Available Start Time Yielding
    CONSERVATIVE = 'conservative'
