"""
Engine-independent snapshots of one simulation instant for the Textual UI.

``SnapshotBuilder`` is the only UI module that touches engine internals. It is called from the
simulation thread at most ``ui_hz`` times per real second (never every simulated second) and
produces a ``UISnapshot`` made of plain Python values and numpy arrays, so the views never read
engine state that the simulation is concurrently mutating.
"""
import re
import time as time_module
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from raps.ui.binning import FREE, BUSY, DOWN
from raps.utils import convert_seconds_to_hhmmss

HISTORY_LEN = 240        # samples kept for the sparklines
MAX_JOB_ROWS = 5000      # running + queued jobs shipped per snapshot
_FACILITY_KEY = re.compile(r"^simulator\[1\]\.(?P<path>.+)\.summary\.(?P<name>\w+)$")
_CDU_PREFIX = "simulator[1].datacenter[1].computeBlock[{i}].cdu[1].summary."


def _nominal(x):
    """Strip the uncertainty off a ufloat; pass everything else through."""
    return getattr(x, "nominal_value", x)


def to_float_array(a):
    """float64 ndarray from an array that may hold ufloats (the --uncertainties path)."""
    a = np.asarray(a)
    if a.dtype == object:
        a = np.vectorize(_nominal, otypes=[float])(a)
    return a.astype(np.float64, copy=False)


@dataclass
class UIMeta:
    """Static facts about the run, computed once."""
    system_name: str
    shape: tuple                 # (NUM_CDUS, RACKS_PER_CDU, NODES_PER_RACK) as configured
    total_nodes: int
    nodes_per_rack: int
    num_cdus: int
    racks_per_cdu: int
    missing_nodes: int
    has_cooling: bool
    cooling_backend: Optional[str]   # "fmu" or "surrogate"
    facility_groups: dict            # group name -> sorted short names present in the cooling outputs
    cdu_labels: dict                 # FMU_COLUMN_MAPPING: per-CDU key -> label
    has_network: bool
    topology: Optional[str]
    timestep_start: int
    timestep_end: int
    downscale: int
    power_cdu_w: float
    max_node_power_w: float
    ui_overrides: dict = field(default_factory=dict)  # optional `ui:` block of the system config


@dataclass
class CoolingSnapshot:
    cdu: dict                # per-CDU key (e.g. "T_sec_r_C") -> float array [num_cdus]
    facility: dict           # group -> {short name: value}, empty when the model has no plant
    pue: Optional[float]
    extrapolating: bool      # surrogate inputs were outside its training range


@dataclass
class NetworkSnapshot:
    avg_tx: float
    avg_rx: float
    avg_util: float
    avg_slowdown: float
    congestion_mean: Optional[float]
    link_stats: Optional[dict]          # engine.last_link_stats (max/mean/min/std_dev/top_links/histogram)
    group_matrix: Optional[np.ndarray]  # dragonfly group x group max util seen among the top links
    jobs: list                          # (id, nodes, slowdown, dilated) of the most slowed-down jobs


@dataclass
class UISnapshot:
    meta: UIMeta
    stale: bool                  # reuses the previous tick's power/cooling results
    timestep: int
    time_str: str
    sim_elapsed: float           # simulated seconds since the start of the run
    progress: float              # 0..1
    rate_text: str               # simulated seconds per real second
    banner: Optional[tuple]      # (text, rich style) from SimulationState
    n_running: int
    n_queued: int
    n_completed: int
    n_killed: int
    active_nodes: int
    free_nodes: int
    down_nodes: int
    system_util: float
    node_state: np.ndarray       # uint8 [total_nodes]: FREE / BUSY / DOWN
    node_job: np.ndarray         # int64 [total_nodes]: id of the job on the node, -1 if none
    node_power: np.ndarray       # float32 [total_nodes] in W
    rack_power: np.ndarray       # float64 [num_cdus, racks_per_cdu] in kW
    rack_loss: np.ndarray        # float64 [num_cdus, racks_per_cdu] in kW
    cdu_power: np.ndarray        # float64 [num_cdus] kW (rack sum)
    cdu_loss: np.ndarray         # float64 [num_cdus] kW
    total_power_mw: float
    total_loss_mw: float
    p_flops: Optional[float]
    g_flops_w: Optional[float]
    jobs: list                   # (id, name, account, state, nodes, time_limit, run_time, slowdown, dilated)
    jobs_truncated: bool
    cooling: Optional[CoolingSnapshot]
    network: Optional[NetworkSnapshot]
    history: dict                # name -> list[float] ring buffers for the sparklines


class SnapshotBuilder:
    """Builds ``UISnapshot`` objects from an engine and its per-second ``TickData``."""

    def __init__(self, engine, sim_config=None):
        self.engine = engine
        self.config = engine.config
        self.encrypt = bool(getattr(sim_config, "encrypt", False))
        c = self.config
        self.total_nodes = int(c["TOTAL_NODES"])
        self.multitenant = bool(c.get("multitenant", False))
        self._last_power_df = None
        self._rate_last = None
        self._rate_ema = None
        self._rate_text = "..."
        self._history = {k: deque(maxlen=HISTORY_LEN)
                         for k in ("power", "util", "pue", "net_util", "slowdown", "congestion")}
        self._fmu_keys = None   # (key, [full output keys per CDU]) cached on first cooling snapshot
        self._facility_keys = None
        self._facility_nkeys = -1
        self.meta = self._build_meta()

    # -- static -----------------------------------------------------------------------------
    def _build_meta(self):
        e, c = self.engine, self.config
        cm = e.cooling_model
        backend = None
        if cm is not None:
            backend = "surrogate" if type(cm).__name__ == "SurrogateCoolingModel" else "fmu"
        system_config = c.get("system_config")
        overrides = {}
        if system_config is not None and getattr(system_config, "ui", None) is not None:
            overrides = system_config.ui.model_dump(exclude_none=True)
        peak = 0.0
        try:
            from raps.power import compute_node_power
            peak = float(compute_node_power(c.get("CPUS_PER_NODE", 0), c.get("GPUS_PER_NODE", 0), 0, c)[0])
        except Exception:
            pass
        return UIMeta(
            system_name=c["system_name"],
            shape=tuple(int(x) for x in c["SC_SHAPE"]),
            total_nodes=self.total_nodes,
            nodes_per_rack=int(c["NODES_PER_RACK"]),
            num_cdus=int(c["NUM_CDUS"]),
            racks_per_cdu=int(c["RACKS_PER_CDU"]),
            missing_nodes=len(c.get("DOWN_NODES", [])),
            has_cooling=cm is not None,
            cooling_backend=backend,
            facility_groups={},
            cdu_labels=dict(c.get("FMU_COLUMN_MAPPING", {})) if cm is not None else {},
            has_network=bool(e.simulate_network),
            topology=c.get("TOPOLOGY") if e.simulate_network else None,
            timestep_start=int(e.timestep_start),
            timestep_end=int(e.timestep_end),
            downscale=int(e.downscale),
            power_cdu_w=float(c.get("POWER_CDU", 0.0)),
            max_node_power_w=peak,
            ui_overrides=overrides,
        )

    # -- per-snapshot -----------------------------------------------------------------------
    def build(self, tick):
        """Build a UISnapshot from a TickData. Call at most ui_hz times per real second."""
        e, meta = self.engine, self.meta
        stale = tick.power_df is self._last_power_df
        self._last_power_df = tick.power_df

        node_state = self._node_state()
        node_job = self._node_job(tick.running)
        node_power = self._node_power()

        rack_power, rack_loss, cdu_power, cdu_loss = self._power_arrays(tick.power_df)
        total_power_kw = float(cdu_power.sum()) + meta.num_cdus * meta.power_cdu_w / 1000.0
        total_loss_kw = float(cdu_loss.sum())

        jobs, truncated = self._job_rows(tick)
        cooling = self._cooling(tick) if meta.has_cooling and tick.fmu_outputs else None
        network = self._network(tick) if meta.has_network else None

        t_s = tick.current_timestep // meta.downscale
        if t_s < 946684800:
            time_str = convert_seconds_to_hhmmss(t_s)
        else:  # a unix timestamp
            time_str = datetime.fromtimestamp(t_s).strftime("%Y-%m-%d %H:%M")
        sim_elapsed = (tick.current_timestep - meta.timestep_start) / meta.downscale
        span = max(meta.timestep_end - meta.timestep_start, 1)
        progress = min(max((tick.current_timestep - meta.timestep_start) / span, 0.0), 1.0)

        if not stale or not self._history["power"]:
            self._history["power"].append(total_power_kw / 1000.0)
            self._history["util"].append(float(tick.system_util or 0.0))
            if cooling is not None and cooling.pue is not None:
                self._history["pue"].append(cooling.pue)
            if network is not None:
                self._history["net_util"].append(network.avg_util * 100.0)
                self._history["slowdown"].append(network.avg_slowdown)
                self._history["congestion"].append(network.congestion_mean or 0.0)

        state = getattr(e, "sim_state", None)
        return UISnapshot(
            meta=meta,
            stale=stale,
            timestep=int(tick.current_timestep),
            time_str=time_str,
            sim_elapsed=sim_elapsed,
            progress=progress,
            rate_text=self._realtime_rate(t_s),
            banner=state.banner() if state is not None else None,
            n_running=len(tick.running),
            n_queued=len(tick.queue),
            n_completed=int(e.jobs_completed),
            n_killed=int(e.jobs_killed),
            active_nodes=int(tick.num_active_nodes),
            free_nodes=int(tick.num_free_nodes),
            down_nodes=len(tick.down_nodes),
            system_util=float(tick.system_util or 0.0),
            node_state=node_state,
            node_job=node_job,
            node_power=node_power,
            rack_power=rack_power,
            rack_loss=rack_loss,
            cdu_power=cdu_power,
            cdu_loss=cdu_loss,
            total_power_mw=total_power_kw / 1000.0,
            total_loss_mw=total_loss_kw / 1000.0,
            p_flops=tick.p_flops,
            g_flops_w=tick.g_flops_w,
            jobs=jobs,
            jobs_truncated=truncated,
            cooling=cooling,
            network=network,
            history={k: list(v) for k, v in self._history.items()},
        )

    def _node_state(self):
        rm, n = self.engine.resource_manager, self.total_nodes
        state = np.full(n, BUSY, dtype=np.uint8)
        if self.multitenant:
            # Same busy rule as Engine.prepare_timestep: any core/GPU in use makes the node busy
            for node in rm.nodes:
                i = node["id"]
                if node["is_down"]:
                    state[i] = DOWN
                elif (node["available_cpu_cores"] == node["total_cpu_cores"]
                      and node["available_gpu_units"] == node["total_gpu_units"]):
                    state[i] = FREE
            return state
        free = np.fromiter(rm.available_nodes, dtype=np.int64, count=len(rm.available_nodes))
        state[free[free < n]] = FREE
        if rm.down_nodes:
            down = np.fromiter(rm.down_nodes, dtype=np.int64, count=len(rm.down_nodes))
            state[down[down < n]] = DOWN
        return state

    def _node_job(self, running):
        node_job = np.full(self.total_nodes, -1, dtype=np.int64)
        for job in running:
            nodes = job.scheduled_nodes
            if nodes is None or len(nodes) == 0:
                continue
            idx = np.asarray(nodes, dtype=np.int64)
            idx = idx[idx < self.total_nodes]
            try:
                node_job[idx] = int(job.id)
            except (TypeError, ValueError):
                node_job[idx] = hash(job.id) & 0x7FFFFFFF
        return node_job

    def _node_power(self):
        # power_state is [NUM_CDUS, RACKS_PER_CDU, NODES_PER_RACK]; C-order ravel is the node id order
        p = to_float_array(self.engine.power_manager.power_state).ravel()
        out = np.zeros(self.total_nodes, dtype=np.float32)
        m = min(len(p), self.total_nodes)
        out[:m] = p[:m]
        return out

    def _power_arrays(self, power_df):
        meta = self.meta
        hdr = self.config["POWER_DF_HEADER"]
        r = meta.racks_per_cdu
        if power_df is None:
            z = np.zeros((meta.num_cdus, r))
            return z, z.copy(), np.zeros(meta.num_cdus), np.zeros(meta.num_cdus)
        vals = to_float_array(power_df[hdr[1:]].to_numpy())
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        # header: CDU, Rack 1..R, Sum, Loss 1..R, Loss
        return vals[:, :r], vals[:, r + 1:2 * r + 1], vals[:, r], vals[:, -1]

    def _job_rows(self, tick):
        ds = self.meta.downscale
        rows = []
        net = self.meta.has_network
        for job in tick.running:
            rows.append(self._job_row(job, ds, net))
            if len(rows) >= MAX_JOB_ROWS:
                break
        for job in tick.queue:
            if len(rows) >= MAX_JOB_ROWS:
                break
            rows.append(self._job_row(job, ds, net))
        truncated = len(tick.running) + len(tick.queue) > len(rows)
        return rows, truncated

    def _job_row(self, job, ds, net):
        return (
            job.id,
            "hidden" if self.encrypt else str(job.name),
            str(job.account),
            job.current_state.value,
            int(job.nodes_required),
            (job.time_limit or 0) // ds,
            job.current_run_time // ds,
            float(getattr(job, "slowdown_factor", 0.0)) if net else 0.0,
            bool(getattr(job, "dilated", False)),
        )

    def _cooling(self, tick):
        out = tick.fmu_outputs
        meta = self.meta
        if self._fmu_keys is None:
            self._fmu_keys = [(k, [_CDU_PREFIX.format(i=i) + k for i in range(1, meta.num_cdus + 1)])
                              for k in meta.cdu_labels]
        cdu = {}
        for k, full in self._fmu_keys:
            cdu[k] = np.array([_nominal(out.get(f, np.nan)) for f in full], dtype=np.float64)
        if self._facility_nkeys != len(out):
            self._facility_nkeys = len(out)
            self._facility_keys = parse_facility_keys(out)
            meta.facility_groups = {}
            for group, name, _ in self._facility_keys:
                meta.facility_groups.setdefault(group, []).append(name)
        facility = {}
        for group, name, key in self._facility_keys:
            facility.setdefault(group, {})[name] = float(_nominal(out[key]))
        pue = out.get("pue")
        return CoolingSnapshot(
            cdu=cdu,
            facility=facility,
            pue=float(_nominal(pue)) if pue is not None else None,
            extrapolating=bool(getattr(self.engine.cooling_model, "extrapolating", False)),
        )

    def _network(self, tick):
        e = self.engine
        stats = e.last_link_stats
        hist = e.net_congestion_history
        cong = float(hist[-1][1]) if hist else None
        jobs = sorted(((j.id, int(j.nodes_required), float(getattr(j, "slowdown_factor", 0.0)),
                        bool(getattr(j, "dilated", False))) for j in tick.running),
                      key=lambda r: r[2], reverse=True)[:50]
        topology = self.meta.topology
        return NetworkSnapshot(
            avg_tx=float(tick.avg_net_tx or 0.0),
            avg_rx=float(tick.avg_net_rx or 0.0),
            avg_util=float(tick.avg_net_util or 0.0),
            avg_slowdown=float(tick.slowdown_per_job or 0.0),
            congestion_mean=cong,
            link_stats=stats,
            group_matrix=dragonfly_group_matrix(stats, self.config) if topology == "dragonfly" else None,
            jobs=jobs,
        )

    def _realtime_rate(self, sim_seconds):
        """Smoothed simulated seconds per wall-clock second, as a string like "3.6k"."""
        now = time_module.monotonic()
        state = getattr(self.engine, "sim_state", None)
        epoch = state.pause_epoch if state is not None else 0
        last = self._rate_last
        if last is None or last[2] != epoch:
            self._rate_last = (now, sim_seconds, epoch)  # first sample, or pause/resume: new baseline
            return self._rate_text
        wall = now - last[0]
        if wall < 0.5:  # too short to measure; keep the previous reading
            return self._rate_text
        inst = (sim_seconds - last[1]) / wall
        prev = self._rate_ema
        self._rate_ema = inst if prev is None else 0.7 * prev + 0.3 * inst
        self._rate_last = (now, sim_seconds, epoch)
        r = self._rate_ema
        if state is not None:
            state.measured_rate = r
        self._rate_text = format_rate(r)
        return self._rate_text


def format_rate(r):
    if r >= 1e6:
        return f"{r / 1e6:.1f}M"
    if r >= 1e3:
        return f"{r / 1e3:.1f}k"
    if r >= 0.1:
        return f"{r:.1f}"
    return f"{r:.2f}"


def parse_facility_keys(outputs):
    """
    Facility (non per-CDU) cooling output keys actually present, as sorted
    (group, short name, full key) tuples. The group is the model component the key belongs to
    (e.g. coolingTowerLoop, hotWaterLoop, datacenter). Per-CDU keys are skipped. The FMU emits
    these; the surrogate only emits per-CDU keys, so its result is empty.
    """
    found = []
    for key in outputs:
        if "computeBlock" in key:
            continue
        m = _FACILITY_KEY.match(key)
        if m:
            group = m.group("path").split(".")[-1].replace("[1]", "")
            found.append((group, m.group("name"), key))
    return sorted(found)


def dragonfly_group_matrix(stats, config):
    """
    Group x group matrix of the highest link utilization among the reported top links, for the
    dragonfly view. Router names are r_{group}_{router}. None if there are no group-level links.
    """
    if not stats or not stats.get("top_links"):
        return None
    n = int(config.get("DRAGONFLY_D", 0)) or 1
    links = []
    for (u, v), util in stats["top_links"]:
        gu, gv = _router_group(u), _router_group(v)
        if gu is not None and gv is not None:
            links.append((gu, gv, util))
            n = max(n, gu + 1, gv + 1)
    if not links:
        return None
    mat = np.zeros((n, n))
    for gu, gv, util in links:
        mat[gu, gv] = mat[gv, gu] = max(mat[gu, gv], util)
    return mat


def _router_group(name):
    parts = str(name).split("_")
    if len(parts) >= 3 and parts[0] == "r" and parts[1].isdigit():
        return int(parts[1])
    return None
