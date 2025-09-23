./main.py run -f ~/data/marconi100/job_table.parquet --system marconi100 --ff 4381000 -t 61000 -o --policy replay
./main.py run -f ~/data/marconi100/job_table.parquet --system marconi100 --ff 4381000 -t 61000 -o --policy fcfs
./main.py run -f ~/data/marconi100/job_table.parquet --system marconi100 --ff 4381000 -t 61000 -o --policy fcfs --backfill easy
./main.py run -f ~/data/marconi100/job_table.parquet --system marconi100 --ff 4381000 -t 61000 -o --policy priority --backfill firstfit
