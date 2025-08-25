python main.py -f ~/data/marconi100/job_table.parquet --system marconi100 --ff 4381000 -t 61000 -o --policy replay
python main.py -f ~/data/marconi100/job_table.parquet --system marconi100 --ff 4381000 -t 61000 -o --policy fcfs
python main.py -f ~/data/marconi100/job_table.parquet --system marconi100 --ff 4381000 -t 61000 -o --policy fcfs --backfill easy
python main.py -f ~/data/marconi100/job_table.parquet --system marconi100 --ff 4381000 -t 61000 -o --policy priority --backfill firstfit
