"""
Terminal user interfaces for RAPS.

- ``classic``: the original single-screen rich ``Live`` layout (``LayoutManager``).
- ``snapshot``/``binning``: ``SnapshotBuilder``/``UISnapshot`` and the pixel-fitting math, an
  engine-independent view of one simulation instant that the Textual UI renders from.
- ``app``/``views``/``widgets``: the multi-view Textual console (``RapsApp``).

Only ``LayoutManager`` and ``resolve_ui`` are imported eagerly; Textual is imported on demand.
"""
import importlib.util
import sys

from raps.ui.classic import LayoutManager

__all__ = ["LayoutManager", "resolve_ui"]


def resolve_ui(sim_config):
    """
    The UI to run: "textual", "classic" or "none". An unset ui defaults to textual when stdout is
    a terminal and textual is installed, otherwise classic. --noui and --debug mean none.
    """
    if sim_config.noui or sim_config.debug or sim_config.ui == "none":
        return "none"
    ui = sim_config.ui
    if ui is None:
        have_textual = importlib.util.find_spec("textual") is not None
        return "textual" if have_textual and sys.stdout.isatty() else "classic"
    if ui == "textual" and importlib.util.find_spec("textual") is None:
        print("textual is not installed; falling back to the classic UI")
        return "classic"
    return ui
