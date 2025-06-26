def get_template_with_topk(topk):
    return """"

Your goal is to deconstruct a math word problem into a generalized template and a Python function to solve it.

Follow these steps carefully:

1-Identify Variables: Find all numbers in the problem that are required for the calculation.

2-Create Python Code: Write a simple, one-line Python function called solve. It must accept the identified variables as lettered arguments (e.g., a, b, ...). The function body should contain only the mathematical expression.

3-Generalize the Question: Rewrite the problem text. Replace each number you identified with a placeholder numbered with the position, like {{0}}, {{1}}, etc. If the final answer appears in the original text, replace it with {{ans}}.

4-List Variables: Create a list of the original numerical values, in the order they appear and are assigned to variables a, b, and so on.

5-Provide the Answer: Calculate and provide the final numerical answer.

The final output MUST be a single JSON object and nothing else.

EXAMPLE
Input Problem:
"question : Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? answer: How many clips did Natalia sell in May? ** Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nHow many clips did Natalia sell altogether in April and May? ** Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"

Correct JSON Output:
{{
  "code": "def solve(a): return a + (a / 2)",
  "question": "Natalia sold {{0}} clips in April. She sold half as many clips in May. How many clips did Natalia sell altogether?",
  "variables": [48.0],
  "answer": 72.0
}}

TASK
Now, process the following problem:

Problem:
{problem}
    
    """