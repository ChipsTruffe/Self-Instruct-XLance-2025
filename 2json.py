import pyarrow.parquet as pq
import json

# Read Parquet
table = pq.read_table('data/gsm8k/gsm8ksocratic.parquet')

# Convert to a list of dictionaries and save
data = table.to_pylist()
with open('data/gsm8k/gsm8ksocratic', 'w') as f:
    for line in data:
        f.write(json.dumps(line) + "\n")