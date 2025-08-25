import pytest
from datetime import timedelta
from raps.utils import parse_td, convert_to_time_unit, infer_time_unit, TIME_UNITS


@pytest.mark.parametrize("input,expected", [
    ("1", timedelta(seconds=1)),
    ("1m", timedelta(minutes=1)),
    (timedelta(minutes=1), timedelta(minutes=1)),
    (2, timedelta(seconds=2)),
    ("PT2S", timedelta(seconds=2)),
])
def test_parse_td(input, expected):
    assert parse_td(input) == expected


def test_parse_td_error():
    with pytest.raises(ValueError):
        parse_td("1x")


@pytest.mark.parametrize("input,unit,expected", [
    ("1s", 's', 1),
    ("1m", 's', 60),
    (0, 'ms', 0),
    (timedelta(seconds=6), 'ms', 6000),
])
def test_convert_to_time_unit(input, unit, expected):
    assert convert_to_time_unit(input, unit) == expected


@pytest.mark.parametrize("input,expected", [
    ("1s", 's'),
    ("1000ms", 'ms'),
    (0, 's'),
    (timedelta(seconds=6), 's'),
    (timedelta(days=6), 's'),
    (timedelta(milliseconds=6), 'ms'),
    (timedelta(milliseconds=60), 'cs'),
])
def test_infer_time_unit(input, expected):
    assert infer_time_unit(input) == TIME_UNITS[expected]
