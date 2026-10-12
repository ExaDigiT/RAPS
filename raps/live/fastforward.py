"""
Fast-forward the live console: play the twin's forecast of the queue as if it were happening now.

``Playback`` turns a ``TwinResult`` (see ``twin.simulate``, which records when each job starts and
ends and on which nodes) into a ``ClusterState`` for any time t seconds from now, so the tabs render
the future exactly as they render the present: jobs start and finish, nodes change hands, power moves.
Like the twin, it has no new submissions, so the machine drains once the queue is done.
"""
from dataclasses import replace
from datetime import datetime
from datetime import timedelta

MAX_SPAN = 7 * 86400   # stop the clock here at the latest

# simulated seconds per real second
SPEEDS = (60, 300, 900, 1800, 3600, 4 * 3600, 12 * 3600)
DEFAULT_SPEED = 3


class Playback:
    def __init__(self, state, res):
        self.state = state    # the live state when the run began (the "now" of t = 0)
        self.res = res
        self.end = min(max((row[0] for row in res.timeline), default=0), MAX_SPAN)

    def state_at(self, t):
        st, res = self.state, self.res
        at = res.at
        jobs = []
        busy = {}   # node -> watts while a simulated job is on it
        for j in st.jobs or []:
            run = res.runs.get(j.id)
            if run is None:
                # not in the simulation (outside the pools, dependency, ...): running jobs age, others wait
                jobs.append(replace(j, elapsed=j.elapsed + int(t)) if j.running else j)
                continue
            s, e, nodes = run
            if e <= t:
                continue   # finished
            if s <= t:
                w = res.job_w.get(j.id)
                for n in nodes:
                    busy[n] = w
                if j.running:
                    jobs.append(replace(j, elapsed=int(t - s)))
                else:
                    jobs.append(replace(j, state="RUNNING", elapsed=int(t - s), start=at + timedelta(seconds=s),
                                        reason="None", nodelist=",".join(nodes)))
            else:
                jobs.append(replace(j, start=None) if j.pending else j)
        node_states = dict(st.node_states or {})
        power = dict(st.node_power) if st.node_power else None
        for pool in res.pool_nodes:
            for n in pool:
                node_states[n] = "allocated" if n in busy else "idle"
                if power is not None:
                    power[n] = (busy[n] or res.idle_w) if n in busy else res.idle_w
        return replace(st, jobs=jobs, node_states=node_states, node_power=power, twin=self.twin_at(t))

    def twin_at(self, t):
        """The twin's forecast as seen from t: the next 24 h panel and start times are measured from there.

        Datetimes shift back by t (and by the age of the simulation), as the views compare them with the real clock. The plan at time
        limits (Launch tab) is not rebased.
        """
        res = self.res
        now = datetime.now()
        shift = res.at - now + timedelta(seconds=t)   # sim clock -> what the views compare with the real clock
        rows = [(0.0, *res._row_at(t))] + [(ts - t, *rest) for ts, *rest in res.timeline if ts > t]
        return replace(res, at=now, timeline=rows, plan=None,
                       starts={k: None if v is None else v - shift for k, v in res.starts.items()
                               if v is None or v > res.at + timedelta(seconds=t)},
                       ends={k: v - shift for k, v in res.ends.items()},
                       power_now=res.power_at(t))


def fmt_offset(t):
    """+1d 03:25:10 style label for a time t seconds from now."""
    t = int(t)
    d, r = divmod(t, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return f"+{d}d {h:02d}:{m:02d}:{s:02d}" if d else f"+{h:02d}:{m:02d}:{s:02d}"


def fmt_speed(sim_per_real):
    for unit, n in (("h", 3600), ("min", 60)):
        if sim_per_real >= n:
            return f"{sim_per_real / n:g} {unit}/s"
    return f"{sim_per_real:g} s/s"
