import csv
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    # For open-source models
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    # For API-based models
    from google import genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


DEFAULT_INPUT_PATH = os.path.join("RIMO", "RIMO-P_parts_expanded_test.csv")
DEFAULT_OUTPUT_PATH = os.path.join("RIMO", "RIMO-P_evaluation_results.csv")
DEFAULT_MODEL = "Qwen/Qwen3-8B"  # Default to open-source model
DEFAULT_MAX_TOKENS = 512
DEFAULT_SLEEP_SEC = 0.5


def build_evaluation_prompt(sub_problems: List[str], num_parts: int) -> str:
    """Build the evaluation prompt with structured output format."""
    prompt_parts = [
        "You are a brilliant mathematical assistant. Your task is to solve a series of related sub-problems.",
        "",
        "**Instructions:**",
        "1. First, review all the sub-problems listed below.",
        "2. Then, solve each sub-problem in sequential order, starting from Part 1.",
        "3. Present your answer for each part clearly under a distinct heading (e.g., 'Solution for Part 1').",
        "4. If a sub-problem is listed as 'None,' you do not need to provide a solution for that part.",
        "",
        f"In this problem, there are {num_parts} sub-problems:",
        ""
    ]
    
    for i, sub_problem in enumerate(sub_problems, 1):
        prompt_parts.append(f"**Sub-Problem {i}:**")
        prompt_parts.append(f"{sub_problem}")
        prompt_parts.append("")
    
    prompt_parts.extend([
        "Now, proceed to solve the sub-problems part-by-part.",
        "",
        "**IMPORTANT:** Use the exact format below for your solutions:",
        "",
        "Solution for Part 1:",
        "[Your solution here]",
        "",
        "Solution for Part 2:",
        "[Your solution here]",
        "",
        "Solution for Part 3:",
        "[Your solution here]",
        "",
        "Solution for Part 4:",
        "[Your solution here]",
        "",
        "If a part is 'None', write 'N/A' for that solution."
    ])
    
    return "\n".join(prompt_parts)


def extract_solutions_from_response(response_text: str, num_parts: int) -> List[str]:
    """Extract solutions from the LLM response using regex patterns."""
    solutions = []
    
    for i in range(1, 5):  # Always check for 4 parts
        if i <= num_parts:
            # Look for "Solution for Part X:" followed by the solution
            pattern = rf"Solution for Part {i}:\s*\n(.*?)(?=\nSolution for Part|\n$|$)"
            match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
            
            if match:
                solution = match.group(1).strip()
                # Clean up common artifacts
                solution = re.sub(r'^[-=*]+\s*', '', solution)  # Remove leading separators
                solution = re.sub(r'\s*[-=*]+$', '', solution)  # Remove trailing separators
                solutions.append(solution)
            else:
                solutions.append("N/A")
        else:
            solutions.append("N/A")
    
    return solutions


def load_model_and_tokenizer(model_name: str):
    """Load the open-source model and tokenizer."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library not available. Install with: pip install transformers torch")
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_with_open_source(model, tokenizer, prompt: str, max_tokens: int) -> str:
    """Generate response using open-source model."""
    # Build chat-style prompt
    user_msg = {
        "role": "user",
        "content": prompt
    }
    
    # Tokenize
    batch = tokenizer.apply_chat_template(
        [user_msg],
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    # Handle both dict and Tensor returns
    if isinstance(batch, torch.Tensor):
        input_ids = batch.to(model.device)
        attention_mask = None
    else:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
    
    # Generate
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            pad_token_id=model.config.eos_token_id,
            eos_token_id=model.config.eos_token_id,
            do_sample=False,
        )
    
    # Strip prompt and decode
    seq = out[0]
    prompt_len = input_ids.shape[1]
    response_ids = seq[prompt_len:].tolist()
    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    
    return response


def generate_with_gemini(prompt: str, model: str) -> str:
    """Generate response using Gemini API."""
    if not HAS_GEMINI:
        raise ImportError("google-genai library not available. Install with: pip install google-genai")
    
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")


def read_expanded_csv(input_path: str) -> List[Dict[str, str]]:
    """Read the expanded CSV with sub-problems."""
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(row)
        return rows


def write_evaluation_csv(output_path: str, results: List[Dict[str, str]]) -> None:
    """Write evaluation results to CSV."""
    fieldnames = [
        "problem_id", "parts", 
        "sub-problem1", "llm_solution1",
        "sub-problem2", "llm_solution2", 
        "sub-problem3", "llm_solution3",
        "sub-problem4", "llm_solution4"
    ]
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def process_row_with_llm(
    row: Dict[str, str], 
    model, 
    tokenizer, 
    model_type: str,
    max_tokens: int
) -> Dict[str, str]:
    """Process a single row with LLM evaluation."""
    problem_id = row.get("problem_id", "")
    parts = int(row.get("number_of_parts", 1))
    
    # Extract sub-problems
    sub_problems = []
    for i in range(1, 5):
        sub_problem = row.get(f"sub-problem{i}", "").strip()
        if sub_problem and sub_problem != "None":
            sub_problems.append(sub_problem)
        else:
            sub_problems.append("None")
    
    # Build prompt
    prompt = build_evaluation_prompt(sub_problems, parts)
    
    # Generate response
    try:
        if model_type == "open_source":
            response = generate_with_open_source(model, tokenizer, prompt, max_tokens)
        elif model_type == "gemini":
            response = generate_with_gemini(prompt, model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Extract solutions
        solutions = extract_solutions_from_response(response, parts)
        
    except Exception as e:
        print(f"Error processing {problem_id}: {e}")
        solutions = ["ERROR"] * 4
    
    # Build result row
    result = {
        "problem_id": problem_id,
        "parts": str(parts),
        "sub-problem1": sub_problems[0],
        "llm_solution1": solutions[0],
        "sub-problem2": sub_problems[1], 
        "llm_solution2": solutions[1],
        "sub-problem3": sub_problems[2],
        "llm_solution3": solutions[2],
        "sub-problem4": sub_problems[3],
        "llm_solution4": solutions[3],
    }
    
    return result


def main(argv: List[str]) -> int:
    input_path = argv[1] if len(argv) > 1 else DEFAULT_INPUT_PATH
    output_path = argv[2] if len(argv) > 2 else DEFAULT_OUTPUT_PATH
    model_name = argv[3] if len(argv) > 3 else DEFAULT_MODEL
    max_tokens = int(argv[4]) if len(argv) > 4 else DEFAULT_MAX_TOKENS
    sleep_sec = float(argv[5]) if len(argv) > 5 else DEFAULT_SLEEP_SEC
    
    # Determine model type
    if model_name.startswith("gemini"):
        model_type = "gemini"
        model = model_name
        tokenizer = None
    else:
        model_type = "open_source"
        model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Read input data
    print(f"Reading expanded CSV: {input_path}")
    rows = read_expanded_csv(input_path)
    print(f"Found {len(rows)} rows to process")
    
    # Process each row
    results = []
    for idx, row in enumerate(rows, 1):
        problem_id = row.get("problem_id", f"row_{idx:05d}")
        print(f"[{idx}/{len(rows)}] Processing {problem_id}...", end=" ", flush=True)
        
        try:
            result = process_row_with_llm(row, model, tokenizer, model_type, max_tokens)
            results.append(result)
            print("done")
        except Exception as e:
            print(f"failed: {e}")
            # Add error row
            error_result = {
                "problem_id": problem_id,
                "parts": "1",
                "sub-problem1": "ERROR",
                "llm_solution1": "ERROR",
                "sub-problem2": "ERROR", 
                "llm_solution2": "ERROR",
                "sub-problem3": "ERROR",
                "llm_solution3": "ERROR",
                "sub-problem4": "ERROR",
                "llm_solution4": "ERROR",
            }
            results.append(error_result)
        
        # Save progress incrementally
        if idx % 10 == 0:  # Save every 10 rows
            write_evaluation_csv(output_path, results)
            print(f"  (Saved progress: {idx} rows)")
        
        time.sleep(sleep_sec)
    
    # Write final results
    write_evaluation_csv(output_path, results)
    print(f"\n✅ Evaluation complete! Results saved to: {os.path.abspath(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
