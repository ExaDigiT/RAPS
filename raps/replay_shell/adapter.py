"""
The seam between the shell and the simulation engine.

Commands only ever call a ``SimAdapter``; they never touch ``Engine`` attributes. That keeps the engine
details in one file, lets the tests use a fake adapter, and is where thread safety is handled: the engine
runs in the ``raps-sim`` thread (see ``RapsApp._sim_worker``) while commands run in the UI thread.

Rules for adapter methods:
- reads copy what they need (``list(engine.queue)``) and return plain ``JobInfo`` records, not ``Job``s
- writes go through a thread-safe engine entry point (``Engine.inject_job``), never ``engine.queue.append``
"""
from dataclasses import dataclass

from raps.job import JobState
from raps.replay_shell.jobscript import JobSpec


@dataclass
class JobInfo:
    id: int
    name: str
    account: str
    state: str            # Slurm-style short state: PD, R, CD, CA, TO, F
    nodes: int
    time_limit: int       # seconds
    elapsed: int          # seconds run so far
    submit_time: int      # simulated seconds
    start_time: int | None
    end_time: int | None
    nodelist: list


@dataclass
class NodeSummary:
    total: int
    allocated: int
    idle: int
    down: int


_STATE = {JobState.PENDING: "PD", JobState.RUNNING: "R", JobState.COMPLETED: "CD", JobState.COMPLETING: "CG",
          JobState.CANCELLED: "CA", JobState.FAILED: "F", JobState.TIMEOUT: "TO"}


def job_info(job):
    return JobInfo(
        id=job.id, name=str(job.name), account=str(job.account or ""),
        state=_STATE.get(job.current_state, "?"), nodes=int(job.nodes_required),
        time_limit=int(job.time_limit or 0), elapsed=int(job.current_run_time or 0),
        submit_time=int(job.submit_time or 0),
        start_time=None if job.start_time is None else int(job.start_time),
        end_time=None if job.end_time is None else int(job.end_time),
        nodelist=list(job.scheduled_nodes or []))


class SimAdapter:
    """What the Slurm-like commands need from a simulation. ``EngineAdapter`` is the real implementation."""

    def now(self) -> int:
        """Current simulated time in seconds (the engine's ``current_timestep``)."""
        raise NotImplementedError

    def submit(self, spec: JobSpec) -> int:
        """Queue a job at the current simulated time and return its id."""
        raise NotImplementedError

    def queue(self) -> list:
        """Pending and running jobs, as ``JobInfo``."""
        raise NotImplementedError

    def finished(self) -> list:
        """Jobs that finished (completed, timed out, cancelled) in this simulation, as ``JobInfo``."""
        raise NotImplementedError

    def cancel(self, job_id: int) -> bool:
        """Cancel a pending or running job. False if there is no such live job."""
        raise NotImplementedError

    def nodes(self) -> NodeSummary:
        raise NotImplementedError


class EngineAdapter(SimAdapter):
    def __init__(self, engine):
        self.engine = engine

    def now(self):
        return int(self.engine.current_timestep)

    def submit(self, spec):
        """TODO(intern): build a ``Job`` from the spec and hand it to ``Engine.inject_job`` (already
        implemented: thread-safe, drained once per tick in ``run_simulation``). Return the job id.

        Suggested shape: ``Job(job_dict(nodes_required=spec.nodes, name=spec.name, account=spec.account,
        id=<fresh id>, cpu_trace=spec.cpu_util, gpu_trace=spec.gpu_util, ntx_trace=0, nrx_trace=0,
        submit_time=self.now(), time_limit=spec.time_limit, start_time=None, end_time=None,
        expected_run_time=<spec.runtime or spec.time_limit>, scheduled_nodes=[]))``; compare
        ``raps.workloads`` and ``raps.live.twin.raps_job``. Replayed telemetry ids can be anything, so pick an
        id above every id in ``engine.jobs``, ``queue`` and ``running`` (and ``Job._id_counter``).

        Then (these are the hard parts, see README tasks 1 and 2): (1) under the replay policy the
        scheduler honours recorded start times and a job with start_time=None may never start; check
        ``Engine.prepare_system_state`` and ``raps/schedulers`` and make injected jobs schedule like a
        normal policy would. (2) The engine's autoshutdown stops the run when the queue is empty, which can
        end the simulation before an injected job is seen; keep it alive while the shell is open.
        (3) Times are in engine seconds; check ``--downscale`` and ``time_delta`` handling.
        (4) ``spec.partition``/``account`` are unused so far.
        """
        raise NotImplementedError

    def queue(self):
        """TODO(intern): ``[job_info(j) for j in list(e.running) + list(e.queue) + list(e.injected_jobs)]``
        (copy the lists first: the engine thread mutates them). Include ``injected_jobs`` so a job shows up
        in squeue before the next tick."""
        raise NotImplementedError

    def finished(self):
        """TODO(intern): ``engine.job_history_dict`` holds ``JobStatistics.__dict__`` rows (raps/job.py)
        for completed and timed-out jobs only. Return ``JobInfo`` for them (and for cancelled jobs once
        ``cancel`` exists) so that ``sacct`` can list them."""
        raise NotImplementedError

    def cancel(self, job_id):
        """TODO(intern): needs an engine entry point like ``inject_job``: a thread-safe
        ``Engine.cancel_job(id)`` that queues the request and, at the next tick (in
        ``prepare_timestep``), removes a queued job from ``engine.queue`` or ends a running one the way
        the killed_jobs loop does (set_idle, resource_manager.free_nodes_from_job, current_state =
        JobState.CANCELLED, job_history_dict entry)."""
        raise NotImplementedError

    def nodes(self):
        """TODO(intern): build a ``NodeSummary`` from ``engine.config["TOTAL_NODES"]`` and
        ``engine.resource_manager`` (``down_nodes``, ``available_nodes``); allocated is what is left."""
        raise NotImplementedError
