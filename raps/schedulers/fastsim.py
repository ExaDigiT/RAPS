import pandas as pd
import sys
import os
import zmq

from ..policy import PolicyType, BackfillType
from raps.telemetry import Telemetry
from ..job import JobState
from raps.sim_config import args
from raps.system_config import get_system_config

# Run with this command:
# python main.py --system kestrel -f ../data/fastsim_jobs_output.parquet --scheduler fastsim --policy priority --start 2024-09-01T00:00 --end 2024-09-15T00:00

class Scheduler():
    """
    FastSim-backed scheduler (strict lockstep via ZeroMQ).

    Protocol (server side is FastSim --serve):
      - INIT                       -> { init_time }
      - GET { t }                  -> { t, running_ids }  (server acks t after reply)
      - END (on shutdown)          -> { ok: true }

    Semantics at engine second t:
      - R_t := authoritative running IDs from FastSim for t
      - started  = R_t - prev_R
        -> stamp start_time=t (once), assign nodes once, mark RUNNING
      - finished = prev_R - R_t
        -> stamp end_time=t (engine will finalize next tick in prepare_timestep)

      running list for this tick = R_t & finished  (so those finishing at t remain
      visible for one more scheduler call; engine completes them on next second).
    """

    def __init__(self, config, resource_manager, **kwargs):
        self.config = config
        self.policy = PolicyType(kwargs.get('policy'))
        self.bfpolicy = BackfillType(kwargs.get('backfill'))
        self.debug = bool(kwargs.get('debug', False))

        # ---- ZeroMQ client ----
        self.endpoint = kwargs.get('plugin_endpoint', 'ipc:///tmp/fastsim.sock')
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

        # INIT handshake: fetch FastSim's init_time (ISO string).
        self.init_time_iso = self._rpc('INIT').get('init_time')

        self.resource_manager = resource_manager

        # Job metadata: id -> Job
        self.jobids_to_jobs = {}
        self.allocated_jobs = set()   # job_ids we have assigned nodes for
        self.prev_running_ids = set() # R_{t-1}

        # Build the Job objects from RAPS Telemetry (needed so ExaDigiT subsystems have objects)
        args_dict = vars(args)
        config = get_system_config(args.system).get_legacy()
        args_dict['config'] = config
        td = Telemetry(**args_dict)

        print("...Now loading jobs to FastSim scheduler.")
        jobs, _, _ = td.load_data(args.replay)
        for job in jobs:
            self.jobids_to_jobs[job.id] = job

        if self.debug:
            print(f"[RAPS-FastSim] Connected to {self.endpoint}; init_time={self.init_time_iso}", file=sys.stderr)

    def _rpc(self, op, **payload):
        """Send a JSON request and return the JSON reply (dict)."""
        try:
            msg = {'op': op}
            msg.update(payload)
            self._sock.send_json(msg)
            rep = self._sock.recv_json()
        except Exception as e:
            raise RuntimeError(f"[RAPS-FastSim] RPC {op} failed: {e}") from e
        if isinstance(rep, dict) and 'error' in rep:
            raise RuntimeError(f"[RAPS-FastSim] RPC {op} error: {rep['error']}")
        return rep

    def _fastsim_running_ids(self, t: int):
        """Blocking call: get authoritative running job IDs for second t."""
        rep = self._rpc('GET', t=int(t))
        rids = rep.get('running_ids', [])
        return set(rids)

    def schedule(self, queue=None, running=None, current_time=None, accounts=None, sorted=False):
        """
        Called by Engine when RAPS detects an event.
        """
        running = running if running is not None else []

        t = int(current_time)

        # Get authoritative running set for second t (blocks until available)
        R_t = self._fastsim_running_ids(t)

        # Diff vs previous second
        started_ids  = R_t - self.prev_running_ids
        finished_ids = self.prev_running_ids - R_t  # these end at t; engine finalizes next tick

        # Handle starts: stamp start_time, assign nodes, mark RUNNING
        for jid in started_ids:
            job = self.jobids_to_jobs.get(jid)
            if job is None:
                if self.debug:
                    print(f"[RAPS-FastSim][WARN] Unknown job id from FastSim: {jid}", file=sys.stderr)
                continue

            # Assign nodes exactly once
            if jid not in self.allocated_jobs:
                self.resource_manager.assign_nodes_to_job(job, t, self.policy)
                self.allocated_jobs.add(jid)

            # FastSim is authoritative
            job.start_time = t
            # IMPORTANT: prevent premature completion by RM’s default behavior
            job.end_time = None # Prevents RAPS from removing job
            job.state = JobState.RUNNING

        # Handle finishes: stamp end_time=t (engine.prepare_timestep next tick completes)
        running.clear()
        for jid in finished_ids:
            job = self.jobids_to_jobs.get(jid)
            if job is not None:
                # overwrite any prior value; FastSim is the source of truth
                # job.end_time = t
                if job.start_time is not None:
                    observed = t - job.start_time
                    if (job.time_limit is None) or (job.time_limit < observed):
                        # This is necessary since RAPS is handling finishing jobs, but schedule is not always
                        # called at every tick, even though the job may have finished in FastSim during that tick.
                        # TODO: Deal with this, because it messes up the end time of some jobs.
                        # print(f"Extending {job.id} runtime {job.time_limit} to match observed {observed} at finish.")
                        job.time_limit = observed
                # print((f"Job {job.id} is finished, start time: {job.start_time}, wall time: {job.time_limit},"  
                #        f"end time: {job.end_time}, at time {t}. With nodes {job.scheduled_nodes}."))
                job.end_time = t
                job.time_limit = t - job.start_time
                running.append(job)

        # Running list reflects exactly FastSim’s R_t
        for jid in R_t:
            job = self.jobids_to_jobs.get(jid)
            if job is not None:
                # defensively ensure state isn’t stuck at COMPLETED
                if job.state != JobState.RUNNING:
                    job.state = JobState.RUNNING
                running.append(job)

        # Update prev
        self.prev_running_ids = R_t

    def end_sim(self):
        # Ask server to stop
        try:
            self._rpc('END')
        except Exception:
            pass