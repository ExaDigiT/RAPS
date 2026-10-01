"""Borrowed cooling model scaling (cooling.heat_scale / cooling.fmu_num_cdus)."""
from types import SimpleNamespace

import numpy as np
import pytest

from raps.cooling import ThermoFluidsModel

pytestmark = pytest.mark.unit

Q = "simulator_1_datacenter_1_computeBlock_{}_cabinet_1_sources_Q_flow_total"
S = "simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary.{}"
PLANT_W = "simulator[1].centralEnergyPlant[1].coolingTowerLoop[1].summary.W_flow_CT_kW"
TOWB = "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_Towb"


def make(**extra):
    return ThermoFluidsModel(NUM_CDUS=2, RACKS_PER_CDU=2, COOLING_EFFICIENCY=1.0,
                             WET_BULB_TEMP=290.0, TEMPERATURE_KEYS=[TOWB], **extra)


def runtime(model, cdu_power):
    return model.generate_runtime_values(np.asarray(cdu_power, float), SimpleNamespace())


def test_unscaled_is_identity():
    m = make()
    rv = runtime(m, [100.0, 300.0])
    assert rv == {Q.format(1): 50.0, Q.format(2): 150.0, TOWB: 290.0}
    out = {S.format(1, "V_flow_prim_GPM"): 10.0, PLANT_W: 4.0}
    assert m.to_system_frame(dict(out)) == out
    assert m.pue_inputs(rv) is rv


def test_heat_scaled_and_filler_cdus_carry_mean_load():
    m = make(HEAT_SCALE=3.0, FMU_NUM_CDUS=4)
    rv = runtime(m, [100.0, 300.0])                 # per cabinet 50, 150 -> x3 = 150, 450
    assert rv[Q.format(1)] == 150.0 and rv[Q.format(2)] == 450.0
    assert rv[Q.format(3)] == rv[Q.format(4)] == 300.0
    assert rv[TOWB] == 290.0
    assert m.pue_inputs(rv) == {Q.format(1): 50.0, Q.format(2): 150.0}


def test_outputs_converted_to_system_frame():
    m = make(HEAT_SCALE=4.0, FMU_NUM_CDUS=4)
    out = {S.format(1, "T_sec_r_C"): 35.0,          # intensive: unchanged
           S.format(1, "p_sec_s_psig"): 50.0,
           S.format(1, "V_flow_sec_GPM"): 400.0,    # CDU flow / heat_scale
           S.format(2, "W_flow_CDUP_kW"): 8.0,
           S.format(3, "T_sec_r_C"): 35.0,          # filler CDU: dropped
           PLANT_W: 100.0}                          # plant: / heat_scale * 2/4
    got = m.to_system_frame(out)
    assert got == {S.format(1, "T_sec_r_C"): 35.0, S.format(1, "p_sec_s_psig"): 50.0,
                   S.format(1, "V_flow_sec_GPM"): 100.0, S.format(2, "W_flow_CDUP_kW"): 2.0,
                   PLANT_W: 12.5}
