from raps.schedulers.default import Scheduler as DefaultScheduler


class Scheduler(DefaultScheduler):
    """
    Scheduler driven by RL agent actions.
    RAPSEnv.step(action) sets env.pending_action,
    then RLScheduler.schedule() reads it and acts.
    """

    def __init__(self, config, policy, resource_manager, env=None, *args, **kwargs):
        super().__init__(config=config, policy=policy, resource_manager=resource_manager, *args, **kwargs)
        self.env = env
        self.pending_action = None

    def schedule(self, queue, running, current_time, **kwargs):
        if not queue or self.pending_action is None:
            return

        action = self.pending_action
        if action >= len(queue):
            return

        job = queue[action]

        # Check feasibility
        if job.nodes_required <= len(self.resource_manager.available_nodes):
            self.place_job_and_manage_queues(job, queue, running, current_time)
        else:
            # Invalid action → skip or log
            if self.config.args.get("debug", False):
                print(f"[t={current_time}] RL chose invalid job {job.id} (needs {job.nodes_required})")

        # Reset action after use
        self.pending_action = None
