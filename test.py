import json

def compute_average_scores(file_path):
    total = 0.0
    count = 0
    
    with open(file_path, 'r') as file:
        for line in file:
            # Parse JSON from each line
            data = json.loads(line)
            
            # Add score to total and increment count
            total += data['avg_similarity_score']
            count += 1
    
    # Calculate average (handle division by zero)
    return total / count if count > 0 else 0.0

if __name__ == "__main__":
    input_file = "data/gpt3_generations/machine_generated_instructionsgpt-4o.jsonl"
    average = compute_average_scores(input_file)
    print(f"Average sim scores: {average:.4f}")