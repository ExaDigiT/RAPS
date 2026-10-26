# replay_shell: a simulated shell for `raps run`

A small shell inside the simulation console. Commands look like Slurm (`sbatch`, `squeue`, ...) but act on
the **simulated** machine; nothing runs on the real system. Files live in a **sandbox** directory.
It is the replay-side counterpart of the Shell tab in `raps/live/` (where commands hit the real Slurm).

## Try it

    raps run --system 40frontiers --ui textual --shell        # press 7 for the Shell tab
    raps run ... --shell --sandbox-dir /some/dir              # default: ~/.raps/sandbox

In the tab:

    echo "#SBATCH -N 4 -t 30:00 -J demo" > demo.sh
    sbatch demo.sh            # Submitted batch job N, appears in squeue and the Jobs tab
    squeue
    sinfo
    help

`sbatch --runtime 10:00 demo.sh` (or `#RAPS --runtime=10:00` in the script) sets how long the simulated job
runs; by default it runs for its whole time limit. Tests: `tests/unit/test_replay_shell.py` and
`test_shell_tab_runs_commands` in `tests/test_ui_app.py`.

## Design

    ShellView (view.py)          Textual tab, only a prompt and a log; added to RapsApp when shell= is given
      -> ReplayShell (registry.py)   shlex parse, > and >> redirection, dispatch, error handling; UI free
           -> Command handlers (commands/fs.py, commands/slurm.py)    handler(ctx, args) -> status
                -> ctx.sandbox  (sandbox.py)   the only way to reach a file
                -> ctx.adapter  (adapter.py)   the only way to reach the engine
                     -> Engine.inject_job()    thread-safe entry point added to raps/engine.py

Rules that keep it safe and testable:

- **Paths**: every user-supplied path goes through `Sandbox.resolve`. It uses `realpath`, so `..`, absolute
  paths elsewhere and symlinks pointing out are all rejected. `~` is the sandbox root. Never call `open()` or
  `Path()` on a raw argument. (Tests: `test_resolve_*`, `test_symlinks`, `test_commands_cannot_escape`.)
- **No subprocesses**: unknown names are "command not found". Every command is Python in this package.
- **Engine access**: commands use only `SimAdapter` (adapter.py). Reads copy data into `JobInfo`; writes go
  through an engine method that is safe to call from another thread. The engine runs in the `raps-sim`
  thread, the shell in the UI thread (`RapsApp._sim_worker`).
- Handlers fail with `CommandError("message")`; the session prints `name: message` and returns status 1.

## What is done

- Sandbox with confinement and tests.
- Registry, dispatcher, redirection; `pwd cd ls cat mkdir echo help exit`.
- `#SBATCH` parsing (`-N/--nodes`, `-t/--time`, `-J/--job-name`, `-A`, `-p`, plus `--runtime`).
- `sbatch` end to end: builds a `Job` and queues it at the current simulated time (tested on a real Engine).
- `squeue`, `sinfo` (basic tables).
- Integration: `--shell` and `--sandbox-dir` flags on `raps run` (sim_config.py, run_sim.py), tab 7 in
  `raps/ui/app.py`, `Engine.inject_job` hook (engine.py, drained once per tick in `run_simulation`).
  The shell follows `<<` restarts (`_restart_worker` repoints the adapter).

## Task list for the intern (most valuable first)

1. **Make injected jobs schedule correctly in every policy** (`EngineAdapter.submit`). With the replay policy
   (`PolicyType.REPLAY`) the scheduler may expect recorded start times; check `Engine.prepare_system_state`,
   `raps/schedulers/` and confirm an `sbatch` job starts when nodes are free. Add a test with a replayed
   workload, not only synthetic.
2. **Keep the simulation alive while the shell is open.** `complete_timestep` stops the run when the queue and
   running list are empty (`autoshutdown`), which can end the run before a job is submitted. Also decide what
   happens at `timestep_end`. See `RapsApp(autoshutdown=...)`.
3. **scancel**: write `Engine.cancel_job(id)` next to `inject_job` (queue the request, apply it in
   `prepare_timestep`: remove from `engine.queue`, or end a running job like the `killed_jobs` loop does:
   `power_manager.set_idle`, `resource_manager.free_nodes_from_job`, state `JobState.CANCELLED`, entry in
   `job_history_dict`), then `EngineAdapter.cancel` and the `_scancel` handler. Un-xfail `test_scancel`.
4. **sacct**: `EngineAdapter.finished()` from `engine.job_history_dict` (`JobStatistics` in raps/job.py),
   then the table in `_sacct`. Un-xfail `test_sacct`.
5. **More `#SBATCH` directives** (jobscript.py): `--partition`, `--account` (accounts: `engine.accounts`),
   `--gres/--gpus-per-node` (use `gpu_units_required`), `--exclusive`, `--dependency`, `--begin`, `--array`,
   `--nodes=MIN-MAX`. Unknown ones are listed in `JobSpec.ignored`; have `sbatch` print a warning for them.
6. **squeue/sinfo options**: `-u`, `-j`, `-t`, `-o`, `--start` (expected start from the scheduler), per
   partition rows, `sinfo -N` node list.
7. **Shell quality**: history (up/down), tab completion (names from `CommandRegistry.names`, paths via
   `Sandbox.resolve`), `;`/`&&`/`|`, `$VAR`, globs, `touch rm cp mv head tail wc grep`, a script editor.
8. **srun**: semantics are undecided (see questions).
9. **Time units**: `--downscale` and `time_delta` change what a "second" means in the engine; check
   `submit`'s times and the `squeue` TIME columns against `UIMeta.downscale` (raps/ui/snapshot.py).
10. Multi-partition runs (`raps run-parts`, `raps/multi_part_engine.py`) are not wired up; `--shell` is
    only in `raps run`.

## Where to look in the engine

| What | Where |
| --- | --- |
| Queue and running lists | `Engine.queue`, `Engine.running` (engine.py `__init__`) |
| Main loop, one pass per simulated second | `Engine.run_simulation` |
| Job arrival into the queue | `add_eligible_jobs_to_queue`, `_drain_injected_jobs` |
| Completion / timeouts | `prepare_timestep`, `complete_timestep` |
| Scheduling | `self.scheduler.schedule(queue, running, now, ...)`, raps/schedulers/ |
| Node allocation | `raps/resmgr/default.py` (`available_nodes`, `assign_nodes_to_job`, `free_nodes_from_job`) |
| Job object and states | raps/job.py (`Job`, `job_dict`, `JobState`, `JobStatistics`) |
| How the UI reads state | raps/ui/snapshot.py (`SnapshotBuilder.build`), raps/ui/app.py |
| A similar "queue a synthetic job" example | `raps_job` in raps/live/twin.py |
| The real-machine shell for comparison | `Shell` in raps/live/views.py, raps/live/slurm.py (parsers for real Slurm output) |

## Open questions for the project owner

- What should `srun` mean in a simulation (a job that starts when scheduled, or an interactive wait)?
- Should the sandbox default be a persistent `~/.raps/sandbox` (current) or a per-run temp directory?
- Should `sbatch` jobs be marked in the Jobs tab and in `--output` results so that they can be told apart from
  the replayed workload?
