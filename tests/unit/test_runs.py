import os
import time
from raps.runs import find_runs, select_runs, parse_duration, runs_prune
from argparse import Namespace
import pytest


def make_run(root, name, age_days, extra=()):
    d = root / name
    d.mkdir(parents=True)
    (d / "sim_config.yaml").write_text("x")
    (d / "snapshot.npz").write_text("x")
    for f in extra:
        (d / f).write_text("x")
    t = time.time() - age_days * 86400
    os.utime(d, (t, t))
    return d


def test_parse_duration():
    assert parse_duration("2d") == 2 * 86400
    with pytest.raises(ValueError):
        parse_duration("2x")


def test_select_filters(tmp_path):
    make_run(tmp_path, "old-stub", 40)
    make_run(tmp_path, "old-full", 40, extra=["stats.out"])
    make_run(tmp_path, "new-stub", 1)
    make_run(tmp_path / "legacy", "legacy-stub", 50)
    (tmp_path / "not-a-run").mkdir()  # no sim_config.yaml: ignored
    runs = find_runs(tmp_path)
    names = lambda rs: sorted(r.path.name for r in rs)  # noqa: E731
    assert names(runs) == ["legacy-stub", "new-stub", "old-full", "old-stub"]
    now = time.time()
    assert names(select_runs(runs, None, None, True, now)) == ["legacy-stub", "new-stub", "old-stub"]
    assert names(select_runs(runs, 30 * 86400, None, False, now)) == ["legacy-stub", "old-full", "old-stub"]
    assert names(select_runs(runs, 30 * 86400, 3, False, now)) == ["legacy-stub"]  # keep 3 newest


def test_prune_requires_yes(tmp_path):
    d = make_run(tmp_path, "old-stub", 40)
    args = Namespace(older_than="30d", keep=None, stubs=False, runs_dir=tmp_path, yes=False)
    runs_prune(args)
    assert d.exists()
    args.yes = True
    runs_prune(args)
    assert not d.exists()
