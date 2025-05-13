batch_dir=data/gpt_test_generations/


engines=("gpt-4.1" "gpt-4.1-mini" "gpt-4.1-nano" "gpt-4o")

for engine in "${engines[@]}"; do
    python self_instruct/bootstrap_instructions.py \
        --batch_dir "${batch_dir}" \
        --num_instructions_to_generate 60 \
        --seed_tasks_path data/gsm8k/gsm8ktrain.jsonl \
        --engine "$engine"
done