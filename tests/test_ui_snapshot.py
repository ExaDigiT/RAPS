import numpy as np
import pytest

from raps.engine import Engine
from raps.sim_config import SingleSimConfig
from raps.ui import binning
from raps.ui.snapshot import (
    SnapshotBuilder, parse_facility_keys, dragonfly_group_matrix, to_float_array,
)

pytestmark = pytest.mark.nodata


@pytest.fixture(scope="module")
def run():
    """A small frontier run advanced a few ticks, with a builder and the last TickData."""
    cfg = SingleSimConfig.model_validate({"system": "frontier", "time": "3m", "noui": True, "seed": 1})
    engine = Engine(cfg)
    builder = SnapshotBuilder(engine, cfg)
    tick = None
    for i, tick in enumerate(engine.run_simulation()):
        if i >= 40:
            break
    return engine, builder, tick


def test_node_arrays_line_up(run):
    engine, builder, tick = run
    snap = builder.build(tick)
    rm, n = engine.resource_manager, engine.config["TOTAL_NODES"]

    assert snap.node_state.shape == (n,)
    free = np.flatnonzero(snap.node_state == binning.FREE)
    down = np.flatnonzero(snap.node_state == binning.DOWN)
    assert sorted(free.tolist()) == sorted(rm.available_nodes)
    assert set(down.tolist()) == set(rm.down_nodes)
    assert (snap.node_state == binning.BUSY).sum() == tick.num_active_nodes
    assert len(free) == tick.num_free_nodes

    # node_job inverts job.scheduled_nodes; busy nodes belong to a running job
    assert (snap.node_job >= 0).sum() == sum(len(j.scheduled_nodes) for j in tick.running)
    for job in tick.running:
        assert (snap.node_job[np.asarray(job.scheduled_nodes)] == job.id).all()
    assert ((snap.node_job >= 0) == (snap.node_state == binning.BUSY)).all()

    # node_power is the power_state in node id order
    assert np.allclose(snap.node_power, engine.power_manager.power_state.ravel()[:n])


def test_power_and_stats(run):
    engine, builder, tick = run
    snap = builder.build(tick)
    m = snap.meta
    assert snap.rack_power.shape == (m.num_cdus, m.racks_per_cdu)
    assert snap.cdu_power.shape == (m.num_cdus,)
    assert np.allclose(snap.rack_power.sum(axis=1), snap.cdu_power)
    assert snap.total_power_mw > 0
    assert snap.n_running == len(tick.running) and snap.n_queued == len(tick.queue)
    assert 0.0 <= snap.progress <= 1.0
    assert len(snap.jobs) == snap.n_running + snap.n_queued
    assert snap.cooling is None and snap.network is None
    assert len(snap.history["power"]) >= 1


def test_stale_flag(run):
    engine, builder, tick = run
    builder.build(tick)
    assert builder.build(tick).stale  # same power_df object as the previous build


@pytest.mark.parametrize("total,per_rack", [(9600, 128), (158976, 368), (384000, 384), (600048, 72), (980, 20)])
@pytest.mark.parametrize("size", [(80, 46), (250, 140), (40, 20)])
def test_binning_conserves_nodes(total, per_rack, size):
    w, h = size
    plan = binning.plan_nodemap(total, per_rack, w, h)
    assert plan.per_bin >= 1 and plan.n_bins >= 1
    assert plan.n_bins <= w * h or not plan.rack_aligned

    rng = np.random.default_rng(0)
    state = rng.integers(0, 3, total).astype(np.uint8)
    power = rng.random(total).astype(np.float32) * 1000
    job = rng.integers(-1, 50, total)
    bins = binning.bin_nodes(state, power, job, plan.per_bin)

    assert bins["count"].sum() == total
    for k in ("free", "busy", "down"):
        assert (bins[k] >= 0).all() and (bins[k] <= 1).all()
    assert np.allclose(bins["free"] + bins["busy"] + bins["down"], 1.0)
    assert len(bins["count"]) == plan.n_bins
    # mean power weighted by bin sizes recovers the total
    assert np.isclose((bins["power"] * bins["count"]).sum(), power.sum(), rtol=1e-4)

    pix = binning.pixel_bin_index(plan, total, per_rack, w, h)
    used = pix[pix >= 0]
    assert len(used) == len(np.unique(used))  # no bin drawn twice
    if plan.rack_aligned:
        assert len(used) == plan.n_bins  # and every bin has a pixel


def test_binning_one_pixel_per_node_when_room():
    plan = binning.plan_nodemap(980, 20, 200, 100)
    assert plan.per_bin == 1 and plan.rack_aligned


def test_colors_shapes():
    state = np.array([0, 1, 2, 1], dtype=np.uint8)
    bins = binning.bin_nodes(state, np.array([0, 5, 0, 9], dtype=np.float32),
                             np.array([-1, 3, -1, 4]), 2)
    for mode in ("state", "job", "power"):
        rgb = binning.bin_colors(bins, mode, 0.0, 10.0)
        assert rgb.shape == (2, 3) and rgb.dtype == np.uint8


FMU_KEYS = {
    "simulator[1].datacenter[1].computeBlock[1].cdu[1].summary.T_sec_r_C": 30.0,
    "simulator[1].datacenter[1].summary.V_flow_bypass_GPM": 1.0,
    "simulator[1].centralEnergyPlant[1].hotWaterLoop[1].summary.n_EHXs": 3.0,
    "simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].summary.n_CTs": 11.0,
    "pue": 1.1,
}


def test_facility_key_parsing_fmu():
    found = parse_facility_keys(FMU_KEYS)
    assert [(g, n) for g, n, _ in found] == [
        ("coolingTowerLoop", "n_CTs"), ("datacenter", "V_flow_bypass_GPM"), ("hotWaterLoop", "n_EHXs")]


def test_facility_key_parsing_surrogate():
    keys = {"simulator[1].datacenter[1].computeBlock[2].cdu[1].summary.T_sec_s_C": 20.0, "pue": 1.05}
    assert parse_facility_keys(keys) == []


def test_dragonfly_group_matrix():
    stats = {"top_links": [(("r_0_1", "r_2_0"), 0.5), (("r_0_0", "r_0_1"), 0.2), (("h_0_0_0", "r_0_0"), 0.9)]}
    m = dragonfly_group_matrix(stats, {"DRAGONFLY_D": 3})
    assert m.shape == (3, 3) and m[0, 2] == m[2, 0] == 0.5 and m[0, 0] == 0.2
    assert dragonfly_group_matrix({"top_links": []}, {}) is None


def test_to_float_array_handles_ufloats():
    from uncertainties import ufloat
    a = np.array([ufloat(1.0, 0.1), ufloat(2.0, 0.2)], dtype=object)
    assert np.allclose(to_float_array(a), [1.0, 2.0])


def test_resolve_ui_and_flags(monkeypatch):
    from raps.ui import resolve_ui
    base = {"system": "frontier"}
    assert resolve_ui(SingleSimConfig.model_validate({**base, "noui": True})) == "none"
    assert resolve_ui(SingleSimConfig.model_validate({**base, "debug": True})) == "none"
    assert resolve_ui(SingleSimConfig.model_validate({**base, "ui": "classic"})) == "classic"
    cfg = SingleSimConfig.model_validate({**base, "ui": "none"})
    assert cfg.noui and resolve_ui(cfg) == "none"  # --ui none implies noui for the engine
    monkeypatch.setattr("sys.stdout.isatty", lambda: False, raising=False)
    assert resolve_ui(SingleSimConfig.model_validate(base)) == "classic"
    monkeypatch.setattr("sys.stdout.isatty", lambda: True, raising=False)
    assert resolve_ui(SingleSimConfig.model_validate(base)) == "textual"
    cfg = SingleSimConfig.model_validate({**base, "ui_hz": 2, "ui_cycle": 7})
    assert cfg.ui_hz == 2 and cfg.ui_cycle == 7


def test_ui_cli_flags():
    import argparse
    from raps.utils import pydantic_add_args
    parser = argparse.ArgumentParser()
    validate = pydantic_add_args(parser, SingleSimConfig)
    cfg = validate(parser.parse_args(["--ui", "classic", "--ui-hz", "2", "--ui-cycle", "10"]))
    assert (cfg.ui, cfg.ui_hz, cfg.ui_cycle) == ("classic", 2.0, 10.0)


def test_nodemap_group_override():
    plan = binning.plan_nodemap(9600, 128, 250, 140, per_bin_override=16)
    assert plan.per_bin == 16 and plan.rack_aligned


def test_system_ui_block():
    from raps.system_config import SystemConfig, get_system_config
    base = get_system_config("frontier").model_dump(mode="json", exclude_unset=True)
    cfg = SystemConfig.model_validate({**base, "ui": {"node_map_group": 8}})
    assert cfg.ui.node_map_group == 8
    assert cfg.get_legacy()["TOTAL_NODES"] == 9600


def test_job_detail_for_selected_job(run):
    engine, builder, tick = run
    assert builder.build(tick).job_detail is None  # nothing selected
    job = tick.running[0]
    builder.selected_job = job.id
    try:
        d = builder.build(tick).job_detail
        assert str(d.id) == str(job.id) and d.state == "R"
        assert d.nodes_required == len(job.scheduled_nodes)
        covered = sum(b - a + 1 for a, b in d.node_ranges)
        assert covered == len(set(job.scheduled_nodes))
        npr = builder.meta.nodes_per_rack
        assert d.racks == sorted({n // npr for n in job.scheduled_nodes})
        assert len(d.power_w) <= 240
        builder.selected_job = "no-such-job"
        assert builder.build(tick).job_detail is None
    finally:
        builder.selected_job = None


def test_size_histogram_buckets():
    from raps.ui.widgets.histogram import bucket_counts, size_buckets, render_hist, fmt_bound
    bounds = size_buckets(9408)
    assert bounds[0] == 1 and bounds[-1] >= 9408
    counts = bucket_counts([1, 2, 3, 4, 5, 8, 9, 9408], len(bounds))
    # buckets are (2^(i-1), 2^i]: 1 | 2 | 3-4 | 5-8 | 9-16 ... 
    assert counts[:5].tolist() == [1, 1, 2, 2, 1]
    assert counts.sum() == 8 and counts[bounds.index(16384)] == 1
    text = render_hist(counts, counts, [fmt_bound(b) for b in bounds], 80, 8)
    assert len(text.plain.splitlines()) == 8
    assert bucket_counts([], 4).tolist() == [0, 0, 0, 0]
