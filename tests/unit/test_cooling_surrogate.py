"""SurrogateCoolingModel time bookkeeping, using a tiny synthetic bundle.

The fake network predicts a constant normalized delta of +1 per step for every
output, so after n strides every output has risen by a known amount. That
makes the stride and history bookkeeping checkable without a trained model.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from raps.cooling_surrogate import SurrogateCoolingModel  # noqa: E402

pytestmark = pytest.mark.unit

N_CDUS = 2
FAST, SLOW, STRIDE = 3, 30, 30
Q = "simulator_1_datacenter_1_computeBlock_{}_cabinet_1_sources_Q_flow_total"
TOWB = "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"
OUT = "simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary.{}"


class OnesDelta(torch.nn.Module):
    """Every head emits a normalized delta of 1.0 per step; counts its calls."""

    def __init__(self, n_fast_out, n_slow_out, k_fast, k_slow):
        super().__init__()
        self.n_fast_out, self.n_slow_out = n_fast_out, n_slow_out
        self.k_fast, self.k_slow = k_fast, k_slow

    def forward(self, u_fast, y_fast, u_slow, y_slow):
        b = u_fast.shape[0]
        return (torch.ones(b, self.k_fast, self.n_fast_out),
                torch.ones(b, self.k_slow, self.n_slow_out))


@pytest.fixture
def model(tmp_path):
    in_cols = [Q.format(i + 1) for i in range(N_CDUS)] + [TOWB]
    dyn = [OUT.format(i + 1, v) for v in ("V_flow_prim_GPM", "T_prim_r_C") for i in range(N_CDUS)]
    fast_idx, slow_idx = list(range(N_CDUS)), list(range(N_CDUS, 2 * N_CDUS))
    k_fast, k_slow = 20, 20
    ts = torch.jit.script(OnesDelta(len(fast_idx), len(slow_idx), k_fast, k_slow))
    ts.save(str(tmp_path / "model.ts"))
    n = len(dyn)
    coef = [[10.0, 0, 0, 0, 0, 0] for _ in range(n)]          # start-up state: 10 everywhere
    bundle = {
        "format_version": 1, "num_cdus": N_CDUS, "input_cols": in_cols, "dynamic_cols": dyn,
        "stride_s": STRIDE,
        "branches": {"fast": {"name": "fast", "rate": FAST, "history": 100},
                     "slow": {"name": "slow", "rate": SLOW, "history": 40}},
        "heads": [{"name": "G_V", "branch": "fast", "rate": FAST, "horizon": k_fast,
                   "persistence": False, "indices": fast_idx},
                  {"name": "G_T", "branch": "slow", "rate": SLOW, "horizon": k_slow,
                   "persistence": False, "indices": slow_idx}],
        "input_norm": {"mean": [0.0] * len(in_cols), "std": [1.0] * len(in_cols)},
        "output_norm": {"mean": [0.0] * n, "std": [1.0] * n},
        # delta scale 0.01 per fast step, 1.0 per slow step
        "delta_scale": {"3": [0.01] * n, "30": [1.0] * n},
        "input_range": {"min": [0.0] * len(in_cols), "max": [1e9] * len(in_cols)},
        "steady_state": {"coef": coef},
    }
    (tmp_path / "bundle.json").write_text(json.dumps(bundle))
    m = SurrogateCoolingModel(NUM_CDUS=N_CDUS, SURROGATE_PATH=str(tmp_path), SURROGATE_WARMUP_S=0,
                              TEMPERATURE_KEYS=[TOWB], WET_BULB_TEMP=290.0,
                              COOLING_EFFICIENCY=1.0, RACKS_PER_CDU=1,
                              W_HTWPs_KEY="x", W_CTWPs_KEY="y", W_CTs_KEY="z")
    m.initialize()
    return m


def run(model, ticks, dt):
    """Drive simulate_cooling like the engine: one call per tick of `dt` seconds."""
    rack_power = np.full((N_CDUS, 2), 100.0)                     # kW, last column = CDU total
    outs = []
    for t in ticks:
        engine = SimpleNamespace(current_timestep=t, time_delta=dt, timestep_start=0,
                                 power_manager=SimpleNamespace(uncertainties=False),
                                 config={"POWER_UPDATE_FREQ": 15})
        _, y = model.simulate_cooling(rack_power=rack_power, engine=engine)
        outs.append(y)
    return outs


def test_one_forward_per_stride_and_closed_loop_accumulation(model):
    outs = run(model, range(0, 300, 15), 15)                    # 300 s of 15 s ticks
    fast = OUT.format(1, "V_flow_prim_GPM")
    slow = OUT.format(1, "T_prim_r_C")
    # t + dt = 300 s is a stride boundary: 10 committed strides from the start state
    assert outs[-1][slow] == pytest.approx(10.0 + 10 * 1.0)
    # fast head: 10 steps of 0.01 per stride
    assert outs[-1][fast] == pytest.approx(10.0 + 10 * 10 * 0.01)
    # mid-stride output (t + dt = 285 s): 9 committed strides plus 5 of 10 fast steps
    assert outs[-2][fast] == pytest.approx(10.0 + 9 * 0.1 + 5 * 0.01)


def test_tick_length_does_not_change_simulated_time(model, tmp_path):
    per_second = run(model, range(0, 300), 1)[-1]
    model2 = SurrogateCoolingModel(**model.config)
    model2.initialize()
    per_15 = run(model2, range(0, 300, 15), 15)[-1]
    for k in per_second:
        if k.endswith("_C") or k.endswith("_GPM"):
            assert per_second[k] == pytest.approx(per_15[k])


def test_gap_between_calls_holds_previous_inputs(model):
    outs = run(model, [0, 15, 90], 15)                          # gap of 60 s before t = 90
    slow = OUT.format(1, "T_prim_r_C")
    # t + dt = 105 s: 3 committed strides, then halfway through the 4th, where the
    # slow head is linearly interpolated toward its next 30 s value
    assert outs[-1][slow] == pytest.approx(10.0 + 3 * 1.0 + 0.5 * 1.0)
