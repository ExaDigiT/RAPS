#!/bin/bash

start_date="2023-09-06"
end_date="2024-03-18"

DPATH="/path/to/data/lake"

# Convert start and end dates to seconds since the epoch
start_sec=$(date -d "$start_date" +%s)
end_sec=$(date -d "$end_date" +%s)

# Loop over dates from start to end
current_sec=$start_sec
while [ $current_sec -le $end_sec ]; do
    # Format the current date as "YYYY-MM-DD" for DATEDIR
    DATEDIR=$(date -d @$current_sec +"%Y-%m-%d")
    DATEDIRS="date=$DATEDIR"

    # Construct the command with the formatted date
    command="python main.py -d -o --plot power loss -f $DPATH/slurm/joblive/$DATEDIRS,$DPATH/jobprofile/jobprofile/$DATEDIRS >& $DATEDIRS.out &"
    sleep 10
    
    # Execute the command
    echo "Executing: $command"
    eval $command
    
    # Increment the current date by one day (86400 seconds)
    current_sec=$(($current_sec + 86400))
done
