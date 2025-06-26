import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, AsyncIterator
import asyncio
from tqdm.asyncio import tqdm as async_tqdm # For async progress bar
import aiofiles
from utils import load_module_functions

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# New imports for structured output
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
dummy_line = { #for tests
  "id": 0,
  "object": "chat.completion",
  "created": 1677654321,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": """
                {
                "code": "def solve(a): return a + a/2",
                "question": "Natalia sold clips to {{a}} of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
                "answer": "How many clips did Natalia sell in May? ** Natalia sold {{a}}/2 clips in May.\\nHow many clips did Natalia sell altogether in April and May? ** Natalia sold {{a}}+{{a}}/2 = {{ans}} clips altogether in April and May.",
                "variables": [48],
                "answer_num": 72
                }  """
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 6,
    "total_tokens": 26
  }
}


class GeneralizedQuestion(BaseModel):
    """Defines the structured output."""
    code: str = Field(..., description="A Python function named 'solve' that takes the variables as arguments and returns the answer.")
    question: str = Field(..., description="The word problem, with variables marked as {{variable_name}}.")
    variables: list[float] = Field(..., description="A list of numerical values for the variables in the question.")
    answer: float = Field(..., description="The final numerical answer to the question.")



def verify_result(result_dict: dict): 
    #Returns a boolean indicating if the python code provided in the answer, when used with the original variables, gives the right result.
    #Also asserts that the result is in the expected 
    try:
        namespace = {}
        exec(result_dict['code'], namespace)
        func = namespace['solve']
        calculated_answer = func(*result_dict['variables'])
        is_correct = calculated_answer == float(result_dict['answer'])
        if not is_correct:
            logger.warning(f"Verification failed for ID {result_dict['id']}. Expected: {result_dict['answer']}, Got: {calculated_answer}")
        return is_correct
    except Exception as e:
        logger.error(f"Error during verification for ID {result_dict.get('id', 'N/A')}: {e}")
        return False

def load_data_utils(dataset_module_path, template_module_path):
    templates = load_module_functions(template_module_path, "template_module", ["template", "choices", "get_template_with_topk"])
    datasets = load_module_functions(dataset_module_path, "dataset_module", ["get_dataset_iterator", "get_prompt"])
    return (*templates, *datasets)

def build_openai_request_payload(prompt: str) -> Dict[str, Any]:
    """Prepares the messages for the OpenAI API request."""
    return {
        "messages": [
            {"role": "system", "content": "You are an expert at solving math word problems. Generate a JSON object that strictly follows the provided schema. Do not include any other text or explanations."},
            {"role": "user", "content": prompt}
        ]
    }


async def single_request(
    client: AsyncOpenAI, 
    payload: Dict[str, Any],
    request_id: Any,
    semaphore: asyncio.Semaphore,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """
    Sends a single asynchronous request to the OpenAI API using instructor
    to get a structured response.
    """
    async with semaphore: # Acquire semaphore to limit concurrency
        try:
            # The response_model=GeneralizedQuestion tells instructor to parse
            # the response into our Pydantic model.
            response = await client.chat.completions.create(
                model=model,
                messages=payload["messages"],
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=GeneralizedQuestion,
                # Forcing JSON mode for models that support it
                response_format={"type": "json_object"},
            )
            return {
                "id": request_id, 
                "success": True, 
                "response": response, # This is a GeneralizedQuestion object
                "status_code": 200
            }
        except ValidationError as e:
            logger.error(f"Request {request_id} failed Pydantic validation: {e}")
            return {"id": request_id, "success": False, "error": f"Validation Error: {e}", "status_code": None}
        except Exception as e:
            # The openai library handles retries, so this is for terminal errors.
            logger.error(f"Request {request_id} encountered an API error: {e}")
            return {"id": request_id, "success": False, "error": f"API Error: {str(e)}", "status_code": getattr(e, 'status_code', None)}


async def process_batch_requests(
    api_key: str,
    base_url: str,
    requests_data: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    concurrency: int,
    timeout_seconds: int = 120,
    max_retries: int = 3
) -> AsyncIterator[Dict[str, Any]]:
    """
    Processes a batch of requests asynchronously using the instructor-patched client,
    yielding results as they complete.
    """
    # Create an instructor-patched OpenAI client.
    # The client will handle retries automatically based on max_retries.
    client = instructor.patch(AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    ))
    
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []
    for req_info in requests_data:
        task = single_request(
            client=client,
            payload=req_info["payload"],
            request_id=req_info["id"],
            semaphore=semaphore,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        tasks.append(task)
    
    # Yield results as they are completed
    for f in asyncio.as_completed(tasks):
        yield await f


def parse_openai_response(api_response: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:

    output_item = {"id": api_response["id"], "success": api_response["success"]} # Include original ID
    
    if not api_response["success"]:
        output_item["error_details"] = api_response.get("error", "Unknown error")
        output_item["status_code"] = api_response.get("status_code")
        output_item["domains"] = None # Or some other default error value
    else:
        try:
            # Assuming the response structure from your synchronous version
            content = api_response["response"]["choices"][0]["message"]["content"].strip()
            if dataset_name in ["nq_open", "triviaqa"]: # Domain parsing specific to these datasets
                domains = [domain.strip() for domain in content.split(",") if domain.strip()]
                output_item["domains"] = domains
            else:
                output_item["raw_response_content"] = content # For other datasets
            output_item["raw_api_response"] = api_response["response"] # Store full API response if needed
        except (KeyError, IndexError, AttributeError) as e:
            logger.warning(f"Error parsing successful response for ID {api_response['id']}: {str(e)}. API Response: {api_response.get('response')}")
            output_item["success"] = False # Mark as failed if parsing fails
            output_item["error_details"] = f"Parsing error: {str(e)}"
            output_item["domains"] = None
    
    # TODO should return id, question, domains only
    # output_item = {}
    return output_item


def post_process(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Verifies API results from the raw results file and prepares them for final output.
    """
    processed = []
    for r in results:
        # We only care about successful, verified results for the final output.
        if r.get("success") and verify_result(r):
            processed.append(r)

    # Sort the final list by ID to restore the original order
    processed.sort(key=lambda x: x["id"])
    return processed



def parse():
    parser = argparse.ArgumentParser(description="Process datasets with OpenAI API using async requests.")

    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset definition module.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--template", type=str, required=True, help="Py config file, should provide a template text, later called by a `apply_template` function")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--api_key", type=str, required=True)
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent requests to OpenAI")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds for each API request")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for a failed request.")
    parser.add_argument("--resume_from_raw_results", type=str, default="")
    parser.add_argument("--topk", type=int, default=3, help="Number of top domains to return")

    return parser.parse_args()


async def main(args):
    
    logger.info(f"Starting processing for dataset: {args.dataset} from {args.dataset_path}")
    logger.info(f"Using model: {args.model}, temperature: {args.temperature}, concurrency: {args.concurrency}")

    (template, choices, get_template_with_topk, get_dataset_iterator, get_prompt) = load_data_utils(args.dataset, args.template)

    # 1. Prepare raw datasets and apply template
    if get_template_with_topk:
        template = get_template_with_topk(args.topk)
        logger.info(f"Using dynamically generated template with topk={args.topk}")
    else:
        logger.info(f"Using static template")
    logger.info(f"Template content: {template}")


    payloads = []
    # original_questions_map is no longer needed as the question is part of the structured output.
    for item in get_dataset_iterator(args.dataset_path):
        prompt = get_prompt(entry_data=item["data"], template=template)
        payload = build_openai_request_payload(prompt=prompt)
        payloads.append({"id": item["id"], "payload": payload})

    if not payloads:
        logger.info("No valid data found in the dataset. Exiting.")
        return
        
    logger.info(f"Prepared {len(payloads)} requests to send.")

    # 2. Process requests asynchronously and write raw results to file
    raw_results = [] # Collect results in memory
    if not args.resume_from_raw_results:
        logger.info(f"Processing requests and writing raw results to {args.output_path}...")
        
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        successful_count = 0
        total_requests = len(payloads)

        api_responses_stream = process_batch_requests(
            api_key=args.api_key,
            base_url=args.base_url,
            requests_data=payloads,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout,
            max_retries=args.max_retries
        )
        
        try:
            async with aiofiles.open(args.output_path, "w", encoding="utf-8") as fout:
                async for api_response in async_tqdm(api_responses_stream, total=total_requests, desc="Processing OpenAI requests"):
                    if api_response["success"]:
                        successful_count += 1
                        # The response is a Pydantic object, convert to dict before writing
                        output_dict = api_response["response"].model_dump()
                        output_dict["id"] = api_response["id"]
                        output_dict["success"] = True
                        raw_results.append(output_dict)
                        await fout.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
                    else:
                        # Log the error response
                        error_dict = {
                            "id": api_response["id"],
                            "success": False,
                            "error_details": api_response.get("error", "Unknown error"),
                            "status_code": api_response.get("status_code")
                        }
                        raw_results.append(error_dict)
                        await fout.write(json.dumps(error_dict, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"An error occurred during processing and writing to file: {e}")
            return
        logger.info(f"All requests processed. Successful: {successful_count}, Failed: {total_requests - successful_count}")

    else:
        logger.info(f"Resuming from cached raw result at {args.resume_from_raw_results}")
        with open(args.resume_from_raw_results, "rt", encoding="utf-8") as fin:
            raw_results = [json.loads(line) for line in fin]


    logger.info("Post-processing final results...")
    # To test with dummy data, uncomment the following line
    # raw_results = [dummy_line] * 5
    # logger.warning("Using dummy results for post-processing.")
    
    results = post_process(raw_results)
    
    # Overwrite the raw results file with the final, post-processed, and sorted data.
    logger.info(f"Saving {len(results)} verified and processed results to {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as fout:
        for result in results:
            fout.write(
                json.dumps(
                    result, ensure_ascii=False
                ) + "\n"
            )

    logger.info(f"Final results saved to {args.output_path}")
    logger.info("Processing complete!")


if __name__ == "__main__":
    asyncio.run(main(parse()))