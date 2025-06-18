# **Google Cluster Trace V2 (2011) Dataset Overview**

Some of this info may be incorrect. Look here for the official documentation:

https://drive.google.com/file/d/0B5g07T_gRDg9Z0lsSTEtTWtpOW8/view?resourcekey=0-cozD56gA4fUDdrkHnLJSrQ

This document provides a summary of the Google Cluster Trace V2 dataset, released in 2011\. This dataset offers insights into the operation of a large-scale production data center and its workload. It's crucial for research in areas like cluster scheduling, resource management, and workload characterization.

## **1\. Dataset Overview**

* **Scale:** The 2011 traces cover a **single production Borg cell (cluster)**.  
* **Machines:** This cluster consisted of approximately **12,500 machines**.  
* **Time Period:** The dataset spans **29 days** of workload data, collected during **May 2011**.  
* **Total Size:** The total compressed size of the full dataset is around **41 GB**.  
* **Format:** All files are provided in **gzipped CSV (.csv.gz)** format.  
* **Anonymization:** The data is heavily anonymized to protect proprietary information. This means specific hardware details (like CPU models or exact core counts) are not provided, and resource values are normalized. User and job identifiers are opaque hashes.

## **2\. Data Sources and File Contents**

The V2 dataset is organized into subdirectories based on event types. Each subdirectory contains multiple gzipped CSV files (part-NNNNN-of-MMMMM.csv.gz). Each of these CSV files **does NOT have a header row**; the first row contains data.

Here's a detailed look at the contents of the core files you've sampled:

### **2.1. job\_events/part-NNNNN-of-MMMMM.csv.gz**

This file contains records for job events. Each row represents an event in the lifecycle of a job.

**Sample head output:**

0,,3418309,0,70s3v5qRyCO/1PCdI6fVXnrW8FU/w+5CKRSa72xgcIo=,3,IHgtoxEBuUTHNbUeVs4hzptMY4n8rZKLbZg+Jh5fNG4=,wAmgn2H74cdoMuSFwJF3NaUEaudVBTZ0/HaNZBwIpEQ=  
0,,3418314,0,70s3v5qRyCO/1PCdI6fVXnrW8FU/w+5CKRSa72xgcIo=,3,L52XDyhi9x9ChmVBZ1qavOFmnzPeVsvQ2QyGmBZcV4s=,ShNjeaoUeqGV2i9WMKEX9HTeuc9K2Fdfovibt7Mp6qI=  
0,,3418319,0,70s3v5qRyCO/1PCdI6fVXnrW8FU/w+5CKRSa72xgcIo=,3,vq0IN3BWEbkDjYgYvkrVyH6OWoUoDwFFf3j/syEZzLA=,1A2GM17AzHRcKJcJet/oIF7FOORyFcAOcUSpR9Fqou8=

**Schema Description:**

| Column Index | Field Name | Description | Data Type (in CSV) | Notes |
| :---- | :---- | :---- | :---- | :---- |
| **0** | time | Time of event (microseconds) | Integer |  |
| **1** | *(missing value)* | Often an empty string. | String |  |
| **2** | job\_ID | Unique ID of the job | Integer |  |
| **3** | event\_type | Type of event: 0=submit, 1=schedule, 2=evict, 3=fail, 4=finish, 5=kill, 6=lost, 7=update, 8=noop. | Integer | Sample shows 0 (submit). |
| **4** | user\_ID | Opaque ID of the user submitting the job | String | Hashed value. |
| **5** | scheduling\_class | 0=non-production, 1=production, 2=free. Values outside this range (like 3 in sample) might indicate an unlisted class or a specific trace artifact. | Integer |  |
| **6** | job\_name | Opaque ID of the job's name | String | Hashed value. |
| **7** | logical\_job\_name | Opaque ID of the logical job name (for grouping related jobs) | String | Hashed value. |
| **8** | number\_of\_tasks | Number of tasks in the job (typically present only on submit events). | Integer | Can be empty if not applicable or derived for specific event types. |
| **9** | CPU\_request | (Normalized) CPU cores requested per task. | Float |  |
| **10** | memory\_request | (Normalized) memory (RAM) requested per task. | Float |  |

### **2.2. machine\_events/part-NNNNN-of-MMMMM.csv.gz**

This file describes events related to machines in the cluster.

**Sample head output:**

0,5,0,HofLGzk1Or/8Ildj2+Lqv0UGGvY82NLoni8+J/Yy0RU=,0.5,0.2493  
0,6,0,HofLGzk1Or/8Ildj2+Lqv0UGGvY82NLoni8+J/Yy0RU=,0.5,0.2493  
0,7,0,HofLGzk1Or/8Ildj2+Lqv0UGGvY82NLoni8+J/Yy0RU=,0.5,0.2493

**Schema Description:**

| Column Index | Field Name | Description | Data Type (in CSV) | Notes |
| :---- | :---- | :---- | :---- | :---- |
| **0** | time | Time of event (microseconds) | Integer |  |
| **1** | machine\_ID | Unique ID of the machine | Integer | IDs are simple integers, but map to opaque IDs in task\_events / task\_usage. |
| **2** | event\_type | Type of event: 0=add, 1=remove, 2=update | Integer | Sample shows 0 (add). |
| **3** | platform\_ID | Opaque string representing the machine's microarchitecture and chipset version | String | Hashed value. Provides insight into hardware heterogeneity without specifics. |
| **4** | CPU\_capacity | (Normalized) Total CPU cores on the machine (e.g., 0.5, 1.0). | Float | Normalized value relative to the largest CPU capacity in the trace (1.0). |
| **5** | memory\_capacity | (Normalized) Total memory (RAM) on the machine. | Float | Normalized value. |

### **2.3. task\_events/part-NNNNN-of-MMMMM.csv.gz**

This file details events related to individual tasks, which are components of jobs.

**Sample head output:**

0,2,3418309,0,4155527081,0,70s3v5qRyCO/1PCdI6fVXnrW8FU/w+5CKRSa72xgcIo=,3,9,,,,  
0,2,3418309,1,329150663,0,70s3v5qRyCO/1PCdI6fVXnrW8FU/w+5CKRSa72xgcIo=,3,9,,,,  
0,,3418314,0,3938719206,0,70s3v5qRyCO/1PCdI6fVXnrW8FU/w+5CKRSa72xgcIo=,3,9,0.125,0.07446,0.0004244,0

**Schema Description:**

| Column Index | Field Name | Description | Data Type (in CSV) | Notes |
| :---- | :---- | :---- | :---- | :---- |
| **0** | time | Time of event (microseconds) | Integer |  |
| **1** | *(missing value)* | Often an empty string. | String |  |
| **2** | job\_ID | Unique ID of the job this task belongs to | Integer |  |
| **3** | task\_index | The index of the task within the job (0-indexed). Uniquely identifies a task when combined with job\_ID. | Integer |  |
| **4** | machine\_ID | ID of the machine where the event occurred (if applicable). This is typically populated when tasks are scheduled or run. Missing implies task not yet assigned to machine. | Integer | This is the opaque machine ID (hashed), distinct from the simple integer machine\_ID in machine\_events but maps to them. |
| **5** | event\_type | Type of task event: 0=submit, 1=schedule, 2=evict, 3=fail, 4=finish, 5=kill, 6=lost, 7=update, 8=noop, 9=assign. | Integer | Sample shows 0 (submit), 2 (evict). |
| **6** | user\_ID | Opaque ID of the user submitting the job. | String | Hashed value. |
| **7** | scheduling\_class | 0=non-production, 1=production, 2=free. Values like 3, 9 might be other classes. | Integer |  |
| **8** | priority | Integer priority from 0 (lowest) to 11 (highest). | Integer |  |
| **9** | CPU\_request | (Normalized) CPU cores requested by this task. | Float | Empty if not applicable for event type. |
| **10** | memory\_request | (Normalized) memory requested by this task. | Float | Empty if not applicable for event type. |
| **11** | disk\_space\_request | (Normalized) disk space requested by this task. | Float | Empty if not applicable for event type. |
| **12** | constraints | (Binary) 0=no constraints, 1=has constraints. | Integer | Empty if not applicable. |

### **2.4. task\_usage/part-NNNNN-of-MMMMM.csv.gz**

This is typically the largest file, containing periodic snapshots of resource usage for running tasks.

**Sample head output:**

600000000,900000000,3418309,0,4155527081,0.001562,0.06787,0.07568,0.001156,0.001503,0.06787,2.861e-06,0.0001869,0.03967,0.0003567,2.445,0.007243,0,1,0  
600000000,900000000,3418309,1,329150663,0.001568,0.06787,0.07556,0.0003195,0.0007,0.06787,5.722e-06,0.0001879,0.03302,0.0009289,2.1,0.005791,0,1,0

**Schema Description:**

| Column Index | Field Name | Description | Data Type (in CSV) | Notes |
| :---- | :---- | :---- | :---- | :---- |
| **0** | start\_time | Start time of the data sample (microseconds) | Integer |  |
| **1** | end\_time | End time of the data sample (microseconds) | Integer | Typically start\_time \+ 300,000,000 (300 seconds or 5 minutes). |
| **2** | job\_ID | Unique ID of the job | Integer |  |
| **3** | task\_index | Index of the task within the job | Integer |  |
| **4** | machine\_ID | ID of the machine where this task ran during the sample period | Integer | Opaque machine ID (hashed). |
| **5** | CPU\_usage\_rate | Normalized average CPU usage rate (cores per second) during the sample. | Float |  |
| **6** | memory\_usage\_avg | Normalized average memory usage. | Float |  |
| **7** | memory\_usage\_max | Normalized maximum memory usage. | Float |  |
| **8** | disk\_I/O\_time\_avg | Normalized average disk I/O time. | Float |  |
| **9** | disk\_I/O\_time\_max | Normalized maximum disk I/O time. | Float |  |
| **10** | CPUs\_allocated | Normalized CPU cores allocated to the task during this sample. | Float |  |
| **11** | memory\_allocated | Normalized amount of memory allocated. | Float |  |
| **12** | sample\_duration | Duration of the sample period (microseconds). | Float | Usually around 300,000,000 (300 seconds). |
| **13-19** | *(unnamed/unknown)* | Additional columns not explicitly documented. | Mixed | These are usually other system metrics or internal flags. You can name them generically if needed. |

