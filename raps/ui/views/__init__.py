"""The switchable views of the Textual console, in key order (1-6)."""
from raps.ui.views.overview import Overview
from raps.ui.views.jobs import Jobs
from raps.ui.views.nodemap import NodeMap
from raps.ui.views.power import Power
from raps.ui.views.cooling import Cooling
from raps.ui.views.network import Network

VIEWS = [Overview, Jobs, NodeMap, Power, Cooling, Network]

__all__ = ["VIEWS", "Overview", "Jobs", "NodeMap", "Power", "Cooling", "Network"]
