from importlib import import_module


def load_scheduler(scheduler_type="default"):
    """Dynamically loads a scheduler by type."""
    module = import_module(f".{scheduler_type}", package="raps.schedulers")
    return getattr(module, "Scheduler")
