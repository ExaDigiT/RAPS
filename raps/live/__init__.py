"""
Live console for the real machine: Slurm queue, my jobs, node map, system state and a shell.

- ``slurm``: run the Slurm/OLCF CLI tools and parse their output into records (no UI code)
- ``forecast``: per-job runtime / power forecasts (the history rule from ``raps.forecast``)
- ``twin``: RAPS start-time forecast: the live queue run forward through RAPS's scheduler
- ``analysis``: derived numbers (utilization, backlog, queue position); where a RAPS forecast plugs in
- ``poller``: background refresh of each source on its own interval, plus a persisted trend
- ``app``/``views``/``widgets``/``screens``: the Textual UI (``LiveApp``)
"""


splash = None  # main.py's Splash while `raps live` starts up


def say(msg):
    """Start-up progress: the splash's current step; an empty message ends the splash (the console is next)."""
    if splash is None:
        return
    if msg:
        splash.set(msg)
    else:
        splash.stop()


def live_add_parser(subparsers):
    """`raps live`: argument parsing only; Textual is imported when the command actually runs."""
    parser = subparsers.add_parser("live", description="""
        Live console for the real machine: the Slurm queue, your jobs (with queue position and
        allocations), a per-node map, reservations, and a built-in shell. Polls Slurm in the background.
    """)
    parser.add_argument("--user", help="Whose jobs are 'my jobs' (default: $USER)")
    parser.add_argument("--tab", default="overview",
                        choices=["overview", "myjobs", "queue", "nodes", "system", "shell", "launch"],
                        help="Tab to open on")
    parser.add_argument("--history-days", type=int, default=7, help="Days of sacct history in My Jobs")
    parser.add_argument("--forecast-days", type=int, default=30, metavar="D",
                        help="Days of every user's finished jobs (sacct -a) behind the runtime forecasts; "
                             "0 turns them off")
    parser.add_argument("--system", default=None,
                        help="RAPS system config (config/<name>.yaml) for the node power model and the node map "
                             "layout; default: the Slurm cluster name if it has a config, else frontier; '' to skip")
    parser.add_argument("--no-twin", action="store_true",
                        help="Do not run the RAPS start-time forecast (a simulation of the queue every 2 min)")
    parser.add_argument("--slow", type=float, default=1.0, metavar="X",
                        help="Multiply every polling interval by X (e.g. 2 to poll half as often)")

    def impl(args):
        say("loading Textual")
        from raps.live.app import run
        run(args)
    parser.set_defaults(impl=impl)
