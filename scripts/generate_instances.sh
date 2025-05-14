batch_dir=data/gpt_test_generations/

python self_instruct/generate_instances.py \
    --batch_dir ${batch_dir} \
    --output_file machine_generated_instances.jsonl \
    --max_instances_to_gen 5 \
    --engine "gpt-4.1" \
    --request_batch_size 5