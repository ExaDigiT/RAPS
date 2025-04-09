from .utils import ValueComparableEnum


class PolicyType(ValueComparableEnum):
    """Supported scheduling policies."""
    REPLAY = 'replay'  # Default is specified in each scheduler!
    FCFS = 'fcfs'
    PRIORITY = 'priority'
    SJF = 'sjf'
    LJF = 'ljf'


class BackfillType(ValueComparableEnum):
    """Supported backfilling policies."""
    NONE = None
    FIRSTFIT = 'firstfit'
    BESTFIT = 'bestfit'
    GREEDY = 'greedy'
    EASY = 'easy'  # Earliest Available Start Time Yielding
    CONSERVATIVE = 'conservative'
