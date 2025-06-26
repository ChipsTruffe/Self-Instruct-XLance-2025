python self_instruct/AsyncCalls.py \
  --dataset data/gsm8k/dataset.py \
  --dataset_path data/gsm8k/gsm8ksocratic.jsonl \
  --template self_instruct/templates/gsm8k_template.py \
  --output_path data/gpt_generations/generalized.jsonl \
  --api_key sk-CGMc2J1rEjWy7V5c23A33fF0055e4491963eF71d28B2AaEf \
  --base_url https://api.xi-ai.cn/v1 \
  --model gpt-4o \
  --concurrency 20 \
  --topk 5 \
  --max_retries 5

# takes gsm8k dataset, transforms each question, and adds a python script to it so that we can use it to generate identical questions with different variables.