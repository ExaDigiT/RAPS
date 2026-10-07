"""Pilot tests for the Textual console, fed by fake snapshots (no engine needed)."""
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("textual")

from raps.engine import SimulationState  # noqa: E402
from raps.ui.app import RapsApp  # noqa: E402
from raps.ui.snapshot import UIMeta, UISnapshot, CoolingSnapshot, NetworkSnapshot, JobDetail  # noqa: E402

pytestmark = pytest.mark.nodata


def make_meta(cooling=False, network=False, cdus=4, racks=3, per_rack=16, topology="fat-tree"):
    return UIMeta(
        system_name="fake", shape=(cdus, racks, per_rack), total_nodes=cdus * racks * per_rack,
        nodes_per_rack=per_rack, num_cdus=cdus, racks_per_cdu=racks, missing_nodes=0,
        has_cooling=cooling, cooling_backend="fmu" if cooling else None,
        facility_groups={},
        cdu_labels={"T_sec_r_C": "Rack Return T (C)", "V_flow_sec_GPM": "Rack Flowrate (gpm)"} if cooling else {},
        has_network=network, topology=topology if network else None,
        timestep_start=0, timestep_end=1000, downscale=1, power_cdu_w=8000.0, max_node_power_w=3000.0,
    )


def make_snapshot(meta, i=0):
    rng = np.random.default_rng(i)
    n = meta.total_nodes
    state = rng.integers(0, 3, n).astype(np.uint8)
    job = np.where(state == 1, rng.integers(0, 6, n), -1).astype(np.int64)
    rack = rng.random((meta.num_cdus, meta.racks_per_cdu)) * 100
    cooling = None
    if meta.has_cooling:
        cooling = CoolingSnapshot(
            cdu={"T_sec_r_C": 30 + rng.random(meta.num_cdus) * 10, "V_flow_sec_GPM": 900 + rng.random(meta.num_cdus)},
            facility={"coolingTowerLoop": {"n_CTs": 4.0, "W_flow_CT_kW": 20.0, "T_fac_ctw_s_C": 20.0,
                                           "T_fac_ctw_r_C": 25.0, "V_flow_ctw_GPM": 6000.0}},
            pue=1.1, extrapolating=False)
    network = None
    if meta.has_network:
        links = [((f"r_{a}_0", f"r_{b}_1"), 0.1 * (a + b + 1)) for a in range(3) for b in range(3)]
        network = NetworkSnapshot(
            avg_tx=1e6, avg_rx=1e6, avg_util=0.2, avg_slowdown=1.3, congestion_mean=0.4,
            link_stats={"max": 0.9, "mean": 0.3, "min": 0.0, "std_dev": 0.1, "top_links": links,
                        "histogram": [5, 4, 3, 2, 1, 1, 0, 0, 0, 1], "num_links": 17},
            group_matrix=np.random.default_rng(1).random((4, 4)),
            jobs=[(j, 2 + j, 1.0 + 0.1 * j, j % 2 == 0) for j in range(5)])
    detail = JobDetail(
        id=2, name="job2", account="acct", state="R", nodes_required=4,
        node_ranges=[(16, 17), (20, 21)], racks=[1], submit_s=-30.0, start_s=10.0, run_s=60.0 + i,
        limit_s=3600.0, power_w=[1000.0 + 10 * k for k in range(30)], power_now_w=1290.0,
        power_avg_w=1145.0, power_peak_w=1290.0, slowdown=1.1, dilated=True)
    jobs = [(j, f"job{j}", "acct", "R" if j < 4 else "PD", 2 + j, 3600, 60 * j, 1.1, j == 2) for j in range(8)]
    return UISnapshot(
        meta=meta, stale=False, timestep=i, time_str="00:00:%02d" % (i % 60), sim_elapsed=float(i),
        progress=min(i / 100, 1.0), rate_text="1.0k", banner=None, n_running=4, n_queued=4,
        n_completed=0, n_killed=0, active_nodes=int((state == 1).sum()), free_nodes=int((state == 0).sum()),
        down_nodes=int((state == 2).sum()), system_util=55.0, node_state=state, node_job=job,
        node_power=(rng.random(n) * 3000).astype(np.float32), rack_power=rack, rack_loss=rack * 0.07,
        cdu_power=rack.sum(axis=1), cdu_loss=rack.sum(axis=1) * 0.07, total_power_mw=1.5, total_loss_mw=0.1,
        p_flops=10.0, g_flops_w=30.0, jobs=jobs, jobs_truncated=False, cooling=cooling, network=network,
        history={k: [1.0, 2.0, 3.0] for k in ("power", "util", "pue", "net_util", "slowdown", "congestion")},
        job_detail=detail,
    )


class SlowSource:
    """Yields snapshots slowly so the app stays alive for the duration of a test."""

    def __init__(self, meta, n=400, delay=0.05):
        self.meta, self.n, self.delay = meta, n, delay

    def __iter__(self):
        import time
        for i in range(self.n):
            time.sleep(self.delay)
            yield make_snapshot(self.meta, i)


def active_view_ids(app):
    return [v.id for v in app.query("#views > View")]


def run_app(meta, test, size=(120, 40), engine=None, **kw):
    async def go():
        app = RapsApp(engine, source=SlowSource(meta), hz=20, **kw)
        async with app.run_test(size=size) as pilot:
            await pilot.pause(0.4)
            await test(app, pilot)
            await pilot.press("q")
        return app
    return asyncio.run(go())


def test_views_hidden_without_cooling_or_network():
    async def check(app, pilot):
        assert active_view_ids(app) == ["view-1", "view-2", "view-3", "view-4"]
        await pilot.press("5")  # cooling does not apply: ignored
        assert app.query_one("#views").current == "view-1"
    run_app(make_meta(), check)


def test_cooling_and_network_views_shown_when_applicable():
    async def check(app, pilot):
        assert active_view_ids(app) == [f"view-{i}" for i in range(1, 7)]
    run_app(make_meta(cooling=True, network=True), check)


def test_view_switching_keys():
    async def check(app, pilot):
        sw = app.query_one("#views")
        for key in "123456":
            await pilot.press(key)
            assert sw.current == f"view-{key}"
        await pilot.press("tab")
        assert sw.current == "view-1"  # wraps
        await pilot.press("shift+tab")
        assert sw.current == "view-6"
    run_app(make_meta(cooling=True, network=True, topology="dragonfly"), check)


def test_node_map_modes_and_drill_down():
    async def check(app, pilot):
        await pilot.press("3")
        nm = app.query_one("#view-3")
        assert nm.mode == 0
        await pilot.press("c")
        assert nm.mode == 1
        await pilot.press("right", "down")
        await pilot.press("enter")
        assert nm.zoomed
        await pilot.pause(0.3)
        assert app.query_one("#nm-rack").row_count == 16  # one row per node of the rack
        await pilot.press("escape")
        assert not nm.zoomed
    run_app(make_meta(), check, size=(120, 50))


def test_jobs_view_filter_and_sort():
    async def check(app, pilot):
        await pilot.press("2")
        await pilot.pause(0.3)
        table = app.query_one("#jobs-table")
        assert table.row_count == 8
        await pilot.press("slash")
        await pilot.press("j", "o", "b", "3")
        await pilot.pause(0.2)
        assert table.row_count == 1
        await pilot.press("escape")
        await pilot.pause(0.2)
        assert table.row_count == 8
        await pilot.press("s")
        assert app.query_one("#view-2").sort_col == 0
    run_app(make_meta(), check)


def test_pause_and_speed_keys_call_simulation_state():
    state = SimulationState(time_delta=1)
    engine = SimpleNamespace(sim_state=state)

    async def check(app, pilot):
        await pilot.press("space")
        assert state.is_paused()
        assert "PAUSED" in state.banner()[0]
        await pilot.press("k")
        assert not state.is_paused()
        await pilot.press("l")
        assert state.get_time_delta() == 2
        await pilot.press("+")
        assert state.get_time_delta() == 4
        await pilot.press("j")
        assert state.get_time_delta() == 2
        await pilot.press("minus")
        assert state.get_time_delta() == 1
    run_app(make_meta(), check, engine=engine)


def test_zero_resets_speed_to_start_value():
    state = SimulationState(time_delta=5)
    engine = SimpleNamespace(sim_state=state)

    async def check(app, pilot):
        await pilot.press("l", "l")
        assert state.get_time_delta() == 20
        await pilot.press("0")
        assert state.get_time_delta() == 5
        assert "RESET SPEED" in state.banner()[0]
        # slower, down past 1x into the wall-clock throttle, then reset clears both
        await pilot.press("j", "j", "j")
        assert state.get_time_delta() == 1
        state.measured_rate = 100.0
        await pilot.press("j")
        assert state.get_target_rate() is not None
        await pilot.press("0")
        assert state.get_time_delta() == 5 and state.get_target_rate() is None
        await pilot.press("0")
        assert "Already at start speed" in state.banner()[0]
    run_app(make_meta(), check, engine=engine)


@pytest.mark.parametrize("width", [60, 80, 120])
def test_speed_banner_fits_status_bar(width):
    state = SimulationState(time_delta=1)
    engine = SimpleNamespace(sim_state=state)

    async def check(app, pilot):
        await pilot.press("l")
        await pilot.pause(0.3)
        text = app.query_one("#statusbar")._text
        assert "SPEED UP: Δt = 2x" in text.plain
        assert text.cell_len <= app.query_one("#statusbar").content_size.width
    run_app(make_meta(), check, size=(width, 30), engine=engine)


def test_quit_before_completion_marks_aborted():
    app = run_app(make_meta(), lambda app, pilot: asyncio.sleep(0))
    assert app.aborted and not app.finished


def test_finishes_with_banner():
    class Short:
        def __iter__(self):
            meta = make_meta()
            return iter([make_snapshot(meta, i) for i in range(3)])

    async def go():
        app = RapsApp(None, source=Short(), hz=20)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.8)
            assert app.finished
            assert "complete" in app._banner()[0]
            await pilot.press("q")
        assert not app.aborted
    asyncio.run(go())


def test_simulation_error_is_captured():
    def boom():
        yield make_snapshot(make_meta(), 0)
        raise RuntimeError("kaput")

    async def go():
        app = RapsApp(None, source=boom(), hz=20)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause(0.8)
            assert isinstance(app.sim_error[0], RuntimeError)
            await pilot.press("q")
    asyncio.run(go())


@pytest.mark.parametrize("size", [(80, 24), (250, 70)])
@pytest.mark.parametrize("cooling,network,topology", [(False, False, "x"), (True, True, "dragonfly"),
                                                      (True, True, "fat-tree")])
def test_every_view_renders_at_size(size, cooling, network, topology):
    meta = make_meta(cooling=cooling, network=network, topology=topology)

    async def check(app, pilot):
        for pos in range(len(app._active)):
            await pilot.press(str(pos + 1))
            await pilot.pause(0.3)
            svg = app.export_screenshot()
            assert "<svg" in svg
    run_app(meta, check, size=size)


def test_help_screen_and_cycle():
    async def check(app, pilot):
        await pilot.press("question_mark")
        await pilot.pause(0.2)
        assert app.screen.__class__.__name__ == "HelpScreen"
        await pilot.press("escape")
        assert app.screen.__class__.__name__ != "HelpScreen"
        before = app._current
        await pilot.pause(1.2)  # --ui-cycle 0.5 advances the view on its own
        assert app._current != before or len(app._active) == 1
    run_app(make_meta(), check, cycle=0.5)


def test_job_detail_modal_opens_updates_and_shows_in_map():
    async def check(app, pilot):
        await pilot.press("2")
        await pilot.pause(0.3)
        table = app.query_one("#jobs-table")
        table.move_cursor(row=2, animate=False)  # job 2 is the one the fake snapshots carry detail for
        await pilot.press("enter")
        await pilot.pause(0.5)
        assert app.screen.__class__.__name__ == "JobDetailScreen"
        assert app.screen.job_id == "2"
        first = str(app.screen.query_one("#jd-summary").render())
        assert "running" in first and "acct" in first and "kW" in first
        assert "Racks" in str(app.screen.query_one("#jd-place").render())
        await pilot.pause(1.0)  # keeps updating and the app behind it keeps polling without errors
        assert app.screen.__class__.__name__ == "JobDetailScreen"
        await pilot.press("m")
        await pilot.pause(0.3)
        assert app.screen.__class__.__name__ != "JobDetailScreen"
        nm = app.query_one("#view-3")
        assert app.query_one("#views").current == "view-3"
        assert nm.cursor == 1 and nm.focus_job == 2
        await pilot.press("escape")
        assert nm.focus_job is None
    run_app(make_meta(), check, size=(120, 50))
