batch_dir=data/gpt3_generations/


python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 60 \
    --seed_tasks_path data/gsm8k/gsm8ktrain.jsonl \
    --engine "gpt-4.1-mini"