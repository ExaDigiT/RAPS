API documentation available at: https://exadigit.github.io/SimulationServer/

# Launch the simulation server

sudo ./scripts/launch_local.sh

# Population some initial simulations, e.g.,

python api_client.py run   --system frontier   --policy default   --start 2024-01-01T00:00:00Z   --end 2024-01-01T05:00:00Z   --scheduler   --scheduler-num-jobs 1000   --scheduler-seed 100   --scheduler-jobs-mode random
{'id': '2gr3nqgbmfapvlwhwzizgzbxr4', 'user': 'unknown', 'system': 'frontier', 'state': 'running', 'error_messages': None, 'start': '2024-01-01T00:00:00Z', 'end': '2024-01-01T00:10:00Z', 'execution_start': '2025-08-15T20:22:13.778590Z', 'execution_end': None, 'progress_date': '2024-01-01T00:00:00Z', 'progress': 0.0, 'config': {'start': '2024-01-01T00:00:00Z', 'end': '2024-01-01T00:10:00Z', 'system': 'frontier', 'scheduler': {'enabled': True, 'down_nodes': [], 'jobs_mode': 'random', 'schedule_policy': 'fcfs', 'reschedule': False, 'jobs': None, 'seed': 100, 'num_jobs': 1000}, 'cooling': {'enabled': False}}}  

# List the simulations

python api_client.py list

#
export BASE_URL="https://myurl.com"
#python get_api_token.py
python api_client.py list
python api_client.py details --id 5rkkb222xnge7c4ba4oxshqeha
