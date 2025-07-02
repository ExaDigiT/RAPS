To generate file indices needed for Damien's reader from full installation of MIT Supercloud dataset:
This will generate: `file_list.csv` and `job_user_date_full.csv`

    python generate_local_metadata.py /lustre/orion/proj-shared/gen150/exadigit/mit_supercloud/datacenter-challenge/202201

To create the npz file that RAPS can use:

    python create_trace.py /lustre/orion/proj-shared/gen150/exadigit/mit_supercloud/datacenter-challenge/202201

Then to run:

    python main.py -f raps/dataloaders/mit_supercloud/data/mit_supercloud_jobs_21_05_2021__22_05_2021.npz --system mit_supercloud
