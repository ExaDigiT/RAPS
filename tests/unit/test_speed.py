"""--speed: start throttled to a multiple of real time (SimConfig.speed -> SimulationState.target_rate)."""
import pytest

from raps.engine import SimulationState
from raps.sim_config import parse_speed


@pytest.mark.parametrize("text,rate", [("1x", 1.0), ("0.5x", 0.5), ("10", 10.0), (" 2X ", 2.0), (4, 4.0), (None, None)])
def test_parse_speed(text, rate):
    assert parse_speed(text) == rate


@pytest.mark.parametrize("text", ["fast", "0x", "-1x", "0.001x"])
def test_parse_speed_rejects(text):
    with pytest.raises(ValueError):
        parse_speed(text)


def test_state_starts_throttled_and_reset_returns_to_it():
    state = SimulationState(time_delta=1, target_rate=1.0)
    assert state.get_target_rate() == 1.0
    state.slow_down()                 # at 1x step: halves the throttle
    assert state.get_target_rate() == 0.5
    state.reset_speed()
    assert state.get_target_rate() == 1.0


def test_speed_up_from_a_throttled_start_can_unthrottle():
    state = SimulationState(time_delta=1, target_rate=1.0)
    state.measured_rate = 3.0         # published by the UI
    state.speed_up()                  # 2x, below the measured speed
    assert state.get_target_rate() == 2.0
    state.speed_up()                  # 4x would exceed what the machine does: unthrottled
    assert state.get_target_rate() is None


def test_default_is_unthrottled():
    assert SimulationState(time_delta=1).get_target_rate() is None
