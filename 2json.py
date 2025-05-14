import pyarrow.parquet as pq
import json

# Read Parquet
table = pq.read_table('data/gsm8k/test-00000-of-00001.parquet')

# Convert to a list of dictionaries and save
data = table.to_pylist()
with open('data/gsm8k/gsm8ktest', 'w') as f:
    for line in data:
        f.write(json.dumps(line) + "\n")