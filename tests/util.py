import os
from pathlib import Path


def find_project_root():
    path = Path(__file__).resolve()
    while not (path / "main.py").exists():
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
