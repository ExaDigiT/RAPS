"""Forecast dataset: history features must only use jobs that ended before a job was submitted."""
import numpy as np
import pandas as pd
import pytest

from raps.forecast.dataset import add_history, clean
from raps.forecast.evaluate import holdout_users, metrics

pytestmark = pytest.mark.nodata


def raw(rows):
    cols = ["JobIDRaw", "User", "Account", "JobName", "Partition", "QOS", "NNodes", "TimelimitRaw",
            "ElapsedRaw", "Submit", "Start", "End", "State", "ExitCode", "ConsumedEnergyRaw", "Priority", "WorkDir"]
    out = []
    for i, (user, name, submit, start, run, energy) in enumerate(rows):
        s = pd.Timestamp(start)
        out.append([str(i), user, "abc123", name, "batch", "normal", "2", "60", str(run), submit, start,
                    (s + pd.Timedelta(seconds=run)).isoformat(), "COMPLETED", "0:0", str(energy), "1", "/tmp"])
    return pd.DataFrame(out, columns=cols)


def test_history_has_no_future_leakage():
    df = add_history(clean(raw([
        ("alice", "sim", "2026-01-01T00:00:00", "2026-01-01T00:00:00", 600, 600 * 2 * 1000),   # ends 00:10
        ("alice", "sim", "2026-01-01T00:05:00", "2026-01-01T00:05:00", 1800, 0),             # ends 00:35
        ("alice", "sim", "2026-01-01T00:20:00", "2026-01-01T00:20:00", 1200, 0),             # ends 00:40
        ("alice", "sim", "2026-01-01T01:00:00", "2026-01-01T01:00:00", 300, 0),              # sees all three
        ("bob", "other", "2026-01-01T02:00:00", "2026-01-01T02:00:00", 300, 0),
    ])))
    df = df.set_index("job_id")
    assert df.loc["0", "uj_n"] == 0 and np.isnan(df.loc["0", "uj_ratio_last"])
    assert df.loc["1", "uj_n"] == 0                 # job 0 was still running at 00:05
    assert df.loc["2", "uj_n"] == 1                 # job 0 had ended by 00:20, job 1 had not
    assert df.loc["2", "uj_ratio_last"] == pytest.approx(600 / 3600)
    assert df.loc["3", "uj_n"] == 3
    assert df.loc["3", "uj_ratio_last"] == pytest.approx(1200 / 3600)   # job 2 ended last (00:40)
    assert df.loc["3", "uj_ratio_mean5"] == pytest.approx((600 + 1800 + 1200) / 3 / 3600)
    assert df.loc["4", "uj_n"] == 0 and df.loc["4", "u_n"] == 0
    assert df.loc["0", "node_power"] == pytest.approx(1000)
    assert df.loc["3", "uj_power_mean5"] == pytest.approx(1000)        # only job 0 had energy


def test_holdout_is_stable_and_metrics():
    users = pd.Series([f"u{i}" for i in range(2000)])
    h = holdout_users(users, 0.1)
    assert 0.07 < h.mean() < 0.13 and (h == holdout_users(users, 0.1)).all()
    y = np.log1p(np.array([600.0, 1200, 3600]))
    m = metrics(y, y)
    assert m["r2_log"] == pytest.approx(1) and m["medae_min"] == 0 and m["within_25pct"] == 1
