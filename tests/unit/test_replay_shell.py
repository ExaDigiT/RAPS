"""The replay shell: sandbox confinement, command dispatch, job scripts, sbatch against a fake adapter."""
import itertools
import os

import pytest

from raps.replay_shell import Sandbox, SandboxError, build_shell
from raps.replay_shell.adapter import JobInfo, NodeSummary, SimAdapter
from raps.replay_shell.commands import build_registry
from raps.replay_shell.jobscript import parse_job_script, parse_time
from raps.replay_shell.registry import ReplayShell

pytestmark = pytest.mark.nodata


# -- sandbox --------------------------------------------------------------------------------------

@pytest.fixture
def box(tmp_path):
    root = tmp_path / "box"
    root.mkdir()
    (root / "sub").mkdir()
    (root / "a.txt").write_text("hello\n")
    return Sandbox(root)


def test_resolve_inside(box):
    assert box.resolve("a.txt") == box.root / "a.txt"
    assert box.resolve("sub/../a.txt") == box.root / "a.txt"
    assert box.resolve("new/file") == box.root / "new" / "file"   # need not exist
    assert box.resolve(str(box.root / "a.txt")) == box.root / "a.txt"   # absolute but inside
    assert box.resolve(".") == box.root


@pytest.mark.parametrize("path", ["..", "../x", "sub/../..", "sub/../../x", "/etc/passwd", "/", "a\0b",
                                  "~/..", "~/../x"])
def test_resolve_rejects_escapes(box, path):
    with pytest.raises(SandboxError):
        box.resolve(path)


def test_sibling_with_same_prefix_is_outside(box):
    sibling = box.root.parent / (box.root.name + "-evil")
    sibling.mkdir()
    with pytest.raises(SandboxError):
        box.resolve(str(sibling / "f"))
    with pytest.raises(SandboxError):
        box.resolve("../" + sibling.name)


def test_symlinks(box, tmp_path):
    outside = tmp_path / "secret"
    outside.mkdir()
    (outside / "f").write_text("x")
    os.symlink(outside, box.root / "link_out")
    os.symlink(outside / "f", box.root / "file_out")
    os.symlink(box.root / "sub", box.root / "link_in")
    os.symlink("../../..", box.root / "sub" / "up")
    for p in ["link_out", "link_out/f", "file_out", "link_out/new", "sub/up/etc"]:
        with pytest.raises(SandboxError):
            box.resolve(p)
    assert box.resolve("link_in") == box.root / "sub"   # a link that stays inside is fine
    os.symlink("nowhere", box.root / "dangling")
    assert box.resolve("dangling") == box.root / "nowhere"


def test_chdir_and_display(box):
    box.chdir("sub")
    assert box.pwd == "~/sub"
    assert box.resolve("../a.txt") == box.root / "a.txt"
    with pytest.raises(SandboxError):
        box.chdir("../..")
    with pytest.raises(SandboxError):
        box.chdir("../a.txt")   # a file
    assert box.pwd == "~/sub"
    box.chdir("~")
    assert box.pwd == "~"
    box.chdir("sub")
    box.chdir("~/sub/..")
    assert box.pwd == "~"


def test_root_must_exist(tmp_path):
    with pytest.raises(SandboxError):
        Sandbox(tmp_path / "missing")
    assert Sandbox(tmp_path / "made", create=True).root.is_dir()


# -- dispatcher and filesystem commands -----------------------------------------------------------

class FakeAdapter(SimAdapter):
    def __init__(self):
        self.specs = []

    def now(self):
        return 0

    def submit(self, spec):
        self.specs.append(spec)
        return 100 + len(self.specs)

    def queue(self):
        return [JobInfo(7, "job", "acct", "PD", 4, 600, 0, 0, None, None, [])]

    def nodes(self):
        return NodeSummary(total=10, allocated=3, idle=6, down=1)


@pytest.fixture
def shell(box):
    return ReplayShell(box, FakeAdapter(), build_registry())


def run(shell, line):
    r = shell.run(line)
    return r.output, r.status


def test_unknown_command_and_blank(shell):
    out, status = run(shell, "bash -c id")
    assert status == 127 and "command not found" in out
    assert run(shell, "") == ("", 0)
    assert run(shell, "# comment") == ("", 0)
    assert run(shell, "echo 'unterminated")[1] == 2


def test_no_subprocess_execution(shell, box):
    for line in ["ls; id", "echo hi | sh", "`id`"]:
        out, _ = run(shell, line)
        assert "uid=" not in out   # `;` and `|` are plain words for now (see the registry TODO)


def test_fs_builtins(shell, box):
    assert run(shell, "pwd") == ("~\n", 0)
    assert run(shell, "mkdir d")[1] == 0
    assert run(shell, "mkdir d")[1] == 1
    assert run(shell, "cd d")[1] == 0
    assert run(shell, "pwd")[0] == "~/d\n"
    assert run(shell, "echo hi there > f.txt")[1] == 0
    assert run(shell, "echo more >> f.txt")[1] == 0
    assert run(shell, "cat f.txt")[0] == "hi there\nmore\n"
    assert "f.txt" in run(shell, "ls")[0]
    assert (box.root / "d" / "f.txt").read_text() == "hi there\nmore\n"
    assert run(shell, "cd")[1] == 0 and run(shell, "pwd")[0] == "~\n"
    assert run(shell, "mkdir -p x/y/z")[1] == 0
    assert run(shell, "cat nope")[1] == 1


def test_commands_cannot_escape(shell, box):
    for line in ["cd ..", "cat /etc/passwd", "ls /", "ls ../..", "mkdir ../evil", "echo x > ../evil",
                 "echo x >> /tmp/evil", "sbatch ../../x.sh"]:
        out, status = run(shell, line)
        assert status == 1 and "outside the sandbox" in out, line
    assert not (box.root.parent / "evil").exists()


def test_redirect_into_directory_fails(shell):
    run(shell, "mkdir d")
    assert run(shell, "echo x > d")[1] == 1


def test_help_and_exit(shell):
    out, _ = run(shell, "help")
    for name in ["sbatch", "squeue", "ls", "exit"]:
        assert name in out
    assert "sbatch [" in run(shell, "help sbatch")[0]
    assert run(shell, "help nope")[1] == 1
    assert shell.run("exit").exit


def test_command_without_simulation(box):
    s = ReplayShell(box, None, build_registry())
    out, status = run(s, "squeue")
    assert status == 1 and "no simulation" in out
    assert run(s, "pwd")[1] == 0


# -- job scripts and sbatch -----------------------------------------------------------------------

@pytest.mark.parametrize("text,seconds", [("10", 600), ("5:30", 330), ("1:02:03", 3723), ("1-0", 86400),
                                          ("1-2:30", 95400), ("2-01:00:05", 176405)])
def test_parse_time(text, seconds):
    assert parse_time(text) == seconds


def test_parse_job_script():
    spec = parse_job_script("#!/bin/bash\n#SBATCH -N 4\n#SBATCH --time=01:00:00 --job-name=foo\n"
                            "#SBATCH --gres=gpu:8\n#RAPS --runtime=10\nsrun x\n#SBATCH -N 99\n")
    assert (spec.nodes, spec.time_limit, spec.name, spec.runtime) == (4, 3600, "foo", 600)
    assert spec.ignored == ["--gres=gpu:8"]   # and the line after the first command is not read


def test_parse_job_script_defaults_and_nodes_range():
    spec = parse_job_script("echo hi\n")
    assert (spec.nodes, spec.name) == (1, "sbatch")
    assert parse_job_script("#SBATCH --nodes=2-4\n").nodes == 2


def test_sbatch_submits_spec(shell):
    run(shell, "echo '#SBATCH -N 3 -t 30 -J mine' > j.sh")
    out, status = run(shell, "sbatch j.sh")
    assert (out, status) == ("Submitted batch job 101\n", 0)
    spec = shell.adapter.specs[0]
    assert (spec.nodes, spec.time_limit, spec.name) == (3, 1800, "mine")
    run(shell, "sbatch -N 8 --runtime 5 j.sh")   # command-line options override the script
    spec = shell.adapter.specs[1]
    assert (spec.nodes, spec.time_limit, spec.runtime) == (8, 1800, 300)


def test_sbatch_errors(shell):
    assert run(shell, "sbatch")[1] == 1
    assert run(shell, "sbatch missing.sh")[1] == 1
    run(shell, "echo '#SBATCH -t xx' > bad.sh")
    assert run(shell, "sbatch bad.sh")[1] == 1
    run(shell, "echo hi > ok.sh")
    assert run(shell, "sbatch --bogus ok.sh")[1] == 1
    assert shell.adapter.specs == []


def test_squeue_and_sinfo(shell):
    out, _ = run(shell, "squeue")
    assert "JOBID" in out and "job" in out and "0:10:00" in out
    out, _ = run(shell, "sinfo")
    assert "allocated" in out and "10" in out


# -- stubs for the intern: flip these when implemented -------------------------------------------

@pytest.mark.xfail(reason="TODO(intern): scancel", strict=True)
def test_scancel(shell):
    assert run(shell, "scancel 7")[1] == 0


@pytest.mark.xfail(reason="TODO(intern): sacct", strict=True)
def test_sacct(shell):
    assert run(shell, "sacct")[1] == 0


@pytest.mark.xfail(reason="TODO(intern): srun semantics", strict=True)
def test_srun(shell):
    assert run(shell, "srun hostname")[1] == 0


# -- against a real engine ------------------------------------------------------------------------

def test_sbatch_into_a_real_engine(tmp_path):
    from raps.engine import Engine
    from raps.sim_config import SingleSimConfig
    engine = Engine(SingleSimConfig.model_validate({"system": "40frontiers", "numjobs": 2, "time": "10m"}))
    sh = build_shell(engine, tmp_path / "sb")
    run(sh, "echo '#SBATCH -N 2 -t 5:00 -J hello' > j.sh")
    out, status = run(sh, "sbatch j.sh")
    assert status == 0
    job_id = int(out.split()[-1])
    assert job_id not in [j.id for j in engine.jobs]
    assert "hello" in run(sh, "squeue")[0]   # visible before the next tick
    for _ in itertools.islice(engine.run_simulation(autoshutdown=False), 20):
        pass
    job = next(j for j in list(engine.running) + list(engine.queue) if j.id == job_id)
    assert job.name == "hello" and job.submit_time == 0 and job.nodes_required == 2
    assert not engine.injected_jobs


def test_default_job_script_is_seeded_and_submittable(tmp_path):
    from raps.replay_shell import build_shell
    from raps.replay_shell.jobscript import parse_job_script
    shell = build_shell(None, tmp_path)
    script = (tmp_path / "job.sh").read_text()
    spec = parse_job_script(script)
    assert spec.nodes == 4 and spec.runtime == 600
    (tmp_path / "job.sh").write_text("edited")
    build_shell(None, tmp_path)
    assert (tmp_path / "job.sh").read_text() == "edited"
    assert "job.sh" in shell.run("ls").output
