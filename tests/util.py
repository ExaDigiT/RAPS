import os
from typing import Any
from pathlib import Path
import shlex
import json
from raps.engine import Engine
from raps.stats import get_stats
from raps.multi_part_engine import MultiPartEngine
from raps.sim_config import SingleSimConfig, MultiPartSimConfig


def find_project_root():
    path = Path(__file__).resolve()
    while not (path / "pyproject.toml").exists():
        if path.parent == path:
            raise RuntimeError("Could not find project root.")
        path = path.parent
    return path


PROJECT_ROOT = find_project_root()
CONFIG_PATH = PROJECT_ROOT / "config"
DATA_PATH = Path(os.getenv("RAPS_DATA_DIR", PROJECT_ROOT / "data")).resolve()

# Maybe usefull but now all systems are listed explicitly!
system_list = [
    entry for entry in os.listdir(CONFIG_PATH)
    if os.path.isfile(os.path.join(CONFIG_PATH, entry, 'system.json'))
]


def requires_all_markers(request, required_markers):
    markexpr = getattr(request.config.option, "markexpr", "")
    selected = set(part.strip() for part in markexpr.split("and"))
    return required_markers.issubset(selected)


def _get_cmd(config, sub_cmd):
    return f"echo {shlex.quote(json.dumps(config))} | python main.py {sub_cmd} - -o none"


def run_engine(sim_config, include_ticks=False) -> tuple[Engine, dict[str, Any]]:
    """
    Run a simulation to completion. Returns the completed Engine and a dict containing the engine
    stats. If include_ticks is True, the dict will also include a list of all the TickDatas (this
    can be very large, especially if cooling is enabled!)
    """
    # Log command to rerun the test manually for debugging convenience
    print(f"Command to reproduce run:\n    {_get_cmd(sim_config, "run")}")

    sim_config = SingleSimConfig.model_validate(sim_config)
    engine = Engine(sim_config)
    gen = engine.run_simulation()

    stats = {
        "tick_count": 0,
        "tick_datas": [] if include_ticks else None,
    }

    for tick in gen:
        stats['tick_count'] += 1
        if include_ticks:
            stats['tick_datas'].append(tick)

    stats.update(get_stats(engine))

    return engine, stats


def run_multi_part_engine(sim_config, include_ticks=False) -> tuple[MultiPartEngine, dict[str, dict[str, Any]]]:
    """
    Run a multi-part simulation to completion. Returns the completed Engine and a dict containing the engine
    stats for each partition. If include_ticks is True, the dicts will also include a list of all the
    TickDatas (this can be very large, especially if cooling is enabled!)
    """
    # Log command to rerun the test manually for debugging convenience
    print(f"Command to reproduce run:\n    {_get_cmd(sim_config, "run-parts")}")

    sim_config = MultiPartSimConfig.model_validate(sim_config)
    multi_engine = MultiPartEngine(sim_config)
    gen = multi_engine.run_simulation()

    stats = {
        "tick_count": 0,
        "tick_datas": [] if include_ticks else None,
        "partitions": {},
    }

    for tick in gen:
        stats['tick_count'] += 1
        if include_ticks:
            stats['tick_datas'].append(tick)

    for partition, engine in multi_engine.engines.items():
        stats['partitions'][partition] = get_stats(engine)

    return multi_engine, stats
