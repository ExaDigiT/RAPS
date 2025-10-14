def continuous_job_generation(*, engine, timestep, jobs):
    # print("if len(engine.queue) <= engine.continuous_workload.args.maxqueue:")
    # print(f"if {len(engine.queue)} <= {engine.continuous_workload.args.maxqueue}:")
    if len(engine.queue) <= engine.continuous_workload.args.maxqueue:
        new_jobs = engine.continuous_workload.generate_jobs().jobs
        jobs.extend(new_jobs)
