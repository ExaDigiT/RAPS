"""
Simulated Slurm commands. They act on the simulation through ``ctx.adapter`` (a ``SimAdapter``); nothing
runs on the real machine and no real Slurm binary is called.

Done: ``sbatch`` (script directives -> queued job), ``squeue`` (simulated queue), ``sinfo`` (node counts).
Stubs for the intern: ``scancel``, ``sacct``, ``srun`` (and the TODOs inside the others).
"""
from raps.replay_shell.jobscript import apply_option, parse_job_script
from raps.replay_shell.registry import CommandError
from raps.utils import convert_seconds_to_hhmmss


def _sbatch(ctx, args):
    """sbatch [--runtime T] [-N n] [-t T] [-J name] script

    Reads the script from the sandbox, takes its ``#SBATCH`` directives, lets options on the command line
    override them (as real sbatch does) and queues the job at the current simulated time.

    TODO(intern): more options on the command line (everything jobscript.apply_option knows already
    works after the flag parsing below is generalised), ``--wrap="cmd"``, ``--test-only``, and
    printing the warnings in ``spec.ignored`` ("sbatch: ignoring unsupported option --gres").
    """
    adapter = ctx.need_sim()
    opts, script, i = [], None, 0
    while i < len(args):
        a = args[i]
        if a.startswith("--") and "=" in a:
            opts.append((a[2:].split("=", 1)[0], a.split("=", 1)[1]))
        elif a in ("-N", "-t", "-J", "-A", "-p", "--nodes", "--time", "--job-name", "--account",
                   "--partition", "--runtime"):
            if i + 1 >= len(args):
                raise CommandError(f"option {a} needs a value")
            opts.append(({"-N": "nodes", "-t": "time", "-J": "job-name", "-A": "account",
                          "-p": "partition"}.get(a, a.lstrip("-")), args[i + 1]))
            i += 1
        elif a.startswith("-"):
            raise CommandError(f"unsupported option {a}")
        else:
            script = a
        i += 1
    if script is None:
        raise CommandError("usage: sbatch [options] script")
    real = ctx.sandbox.resolve(script)
    if not real.is_file():
        raise CommandError(f"{script}: no such file")
    try:
        spec = parse_job_script(real.read_text(errors="replace"))
        for key, value in opts:
            if not apply_option(spec, key, value):
                raise CommandError(f"unsupported option --{key}")
    except ValueError as exc:
        raise CommandError(str(exc))
    if spec.name == "sbatch":
        spec.name = real.name
    if spec.nodes < 1 or spec.time_limit < 1:
        raise CommandError("invalid job request: nodes and time limit must be positive")
    job_id = adapter.submit(spec)
    ctx.out(f"Submitted batch job {job_id}")


def _squeue(ctx, args):
    """squeue: the simulated queue (running first, then pending).

    TODO(intern): -u/--user, -j <id>, -t <states>, -o format strings, --start (use the scheduler's
    expected start), sorting by priority, and the REASON column.
    """
    adapter = ctx.need_sim()
    ctx.out(f"{'JOBID':>8} {'NAME':<16} {'ACCOUNT':<10} {'ST':<3} {'NODES':>6} {'TIME':>10} {'TIME_LIMIT':>10}")
    for j in adapter.queue():
        ctx.out(f"{j.id:>8} {j.name[:16]:<16} {j.account[:10]:<10} {j.state:<3} {j.nodes:>6} "
                f"{convert_seconds_to_hhmmss(j.elapsed):>10} {convert_seconds_to_hhmmss(j.time_limit):>10}")


def _sinfo(ctx, args):
    """sinfo: node counts by state for the simulated machine.

    TODO(intern): per-partition rows, a node list (``-N``), ``-o`` formats, and a draining/reserved state.
    """
    n = ctx.need_sim().nodes()
    ctx.out(f"{'STATE':<10} {'NODES':>6}")
    for state, count in (("allocated", n.allocated), ("idle", n.idle), ("down", n.down)):
        ctx.out(f"{state:<10} {count:>6}")
    ctx.out(f"{'total':<10} {n.total:>6}")


def _scancel(ctx, args):
    """scancel <jobid> ...: cancel simulated jobs.

    TODO(intern): implement ``EngineAdapter.cancel`` (and ``Engine.cancel_job``, see the notes there);
    then this handler is nearly done: parse ids, call the adapter, report "scancel: Invalid job id N".
    Also -u USER, and cancelling a whole array.
    """
    raise NotImplementedError("see EngineAdapter.cancel")


def _sacct(ctx, args):
    """sacct: finished simulated jobs (id, name, state, elapsed, nodes).

    TODO(intern): implement ``EngineAdapter.finished`` and format the rows like sacct; add -j, -S/-E (in
    simulated time), --format, and ``--starttime`` handling.
    """
    raise NotImplementedError("see EngineAdapter.finished")


def _srun(ctx, args):
    """srun: an interactive step. In a simulation there is nothing to attach to, so the sensible meaning
    is "submit a job that runs this command for its time limit and report when it would start".

    TODO(intern): decide the semantics with the user first (see README.md, open questions).
    """
    raise NotImplementedError("semantics undecided")


def register_all(r):
    r.register("sbatch", "submit a job script to the simulated queue", "sbatch [-N n] [-t time] [-J name] "
               "[--runtime time] script")(_sbatch)
    r.register("squeue", "show the simulated queue", "squeue")(_squeue)
    r.register("sinfo", "show simulated node states", "sinfo")(_sinfo)
    r.register("scancel", "cancel a simulated job (not implemented)", "scancel jobid ...")(_scancel)
    r.register("sacct", "show finished simulated jobs (not implemented)", "sacct")(_sacct)
    r.register("srun", "run a command on the simulated machine (not implemented)", "srun [options] command")(_srun)
