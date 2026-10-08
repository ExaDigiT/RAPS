"""
Live console for the real machine: Slurm queue, my jobs, node map, system state and a shell.

- ``slurm``: run the Slurm/OLCF CLI tools and parse their output into records (no UI code)
- ``analysis``: derived numbers (utilization, backlog, queue position); where a RAPS forecast plugs in
- ``poller``: background refresh of each source on its own interval, plus a persisted trend
- ``app``/``views``/``widgets``/``screens``: the Textual UI (``LiveApp``)
"""


def live_add_parser(subparsers):
    """`raps live`: argument parsing only; Textual is imported when the command actually runs."""
    parser = subparsers.add_parser("live", description="""
        Live console for the real machine: the Slurm queue, your jobs (with queue position and
        allocations), a per-node map, reservations, and a built-in shell. Polls Slurm in the background.
    """)
    parser.add_argument("--user", help="Whose jobs are 'my jobs' (default: $USER)")
    parser.add_argument("--tab", default="overview",
                        choices=["overview", "myjobs", "queue", "nodes", "system", "shell"], help="Tab to open on")
    parser.add_argument("--history-days", type=int, default=7, help="Days of sacct history in My Jobs")
    parser.add_argument("--slow", type=float, default=1.0, metavar="X",
                        help="Multiply every polling interval by X (e.g. 2 to poll half as often)")

    def impl(args):
        from raps.live.app import run
        run(args)
    parser.set_defaults(impl=impl)
