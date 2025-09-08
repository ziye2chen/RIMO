import csv
import os
import re
import sys
import time
from typing import Dict, List

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


DEFAULT_INPUT_PATH = os.path.join("RIMO", "RIMO-P.csv")
DEFAULT_OUTPUT_PATH = os.path.join("RIMO", "RIMO-P_solutions_local_sequential.csv")
DEFAULT_MODEL = "mistralai/Mathstral-7B-v0.1"
DEFAULT_SLEEP_SEC = 0.25
DEFAULT_MAX_NEW_TOKENS = 1024


def load_model_and_tokenizer(model_name: str):
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers library not available. Install with: pip install transformers torch")
    print(f"Loading local model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_rows(path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "problem_id", "parts",
        "sub-problem1", "llm_solution1",
        "sub-problem2", "llm_solution2",
        "sub-problem3", "llm_solution3",
        "sub-problem4", "llm_solution4",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_problem_ids_from_input(input_path: str) -> List[str]:
    """Read problem ids from the original dataset, handling BOM-prefixed header."""
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        key = "problem_id"
        if "\ufeffproblem_id" in fieldnames:
            key = "\ufeffproblem_id"
        else:
            for name in fieldnames:
                if name.replace("\ufeff", "") == "problem_id":
                    key = name
                    break
        ids: List[str] = []
        for row in reader:
            ids.append((row.get(key) or "").strip())
        return ids


def fix_output_problem_ids_if_needed(input_path: str, output_path: str) -> None:
    """If output has placeholder problem_id like 'row_00001', replace with ids from input."""
    with open(output_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        out_rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not out_rows:
        return

    placeholder_count = sum(1 for r in out_rows if (r.get("problem_id") or "").startswith("row_"))
    if placeholder_count == 0:
        return

    input_ids = read_problem_ids_from_input(input_path)
    limit = min(len(out_rows), len(input_ids))
    for i in range(limit):
        out_rows[i]["problem_id"] = input_ids[i]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def clamp_parts(value: str) -> int:
    try:
        n = int((value or "").strip() or 1)
    except ValueError:
        n = 1
    return max(1, min(4, n))


def clean_text(text: str) -> str:
    if text is None:
        return ""
    t = text.strip()
    if t.startswith("```"):
        try:
            start = t.index("\n") + 1
            end = t.rindex("```")
            t = t[start:end].strip()
        except ValueError:
            pass
    t = re.sub(r"^\s*#+\s*.*\n", "", t)
    t = re.sub(r"^\s*Solution[^:]*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_subproblem_prompt(
    original_problem: str,
    sub_problem: str,
    step_index: int,
    total_parts: int,
    prev_solution: str,
) -> str:
    context = [
        "You are a careful mathematician. Solve ONLY the current sub-problem.",
        f"This problem has {total_parts} parts; you are solving part {step_index}.",
        "You may treat the statement below as already proved in the previous step:",
        prev_solution if prev_solution else "(None)",
        "",
        "Original problem (for context):",
        original_problem or "",
        "",
        f"Sub-problem {step_index} (solve this part only):",
        sub_problem or "",
        "",
        "Output requirements:",
        "- Provide a rigorous solution for THIS part only.",
        "- Plain text only, one concise paragraph. No headings, no lists.",
        "- Do not restate the problem. Do not refer to other parts.",
    ]
    return "\n".join(context)


def generate_with_local(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    batch = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(batch, torch.Tensor):
        input_ids = batch.to(model.device)
        attention_mask = None
    else:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=model.config.eos_token_id,
            eos_token_id=model.config.eos_token_id,
            do_sample=False,
        )

    seq = out[0]
    prompt_len = input_ids.shape[1]
    response_ids = seq[prompt_len:].tolist()
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return clean_text(text)


def process_all(
    input_path: str,
    output_path: str,
    model_name: str,
    sleep_sec: float,
    max_new_tokens: int,
) -> None:
    rows = load_rows(input_path)
    model, tokenizer = load_model_and_tokenizer(model_name)

    results: List[Dict[str, str]] = []

    for idx, row in enumerate(rows, 1):
        problem_id = (row.get("problem_id") or f"row_{idx:05d}").strip()
        problem_text = (row.get("problem") or "").strip()
        parts = clamp_parts(row.get("number_of_parts", "1"))

        sub_probs: List[str] = []
        for j in range(1, 4 + 1):
            raw = (row.get(f"sub-problem{j}") or "").strip()
            sub_probs.append(raw if raw else "None")

        solutions = ["N/A", "N/A", "N/A", "N/A"]
        prev_solution = ""

        for j in range(1, parts + 1):
            sub_problem = sub_probs[j - 1]
            if sub_problem.lower() == "none":
                solutions[j - 1] = "N/A"
                prev_solution = ""
                continue

            user_prompt = build_subproblem_prompt(problem_text, sub_problem, j, parts, prev_solution)
            try:
                sol = generate_with_local(
                    model,
                    tokenizer,
                    system_prompt="You are a helpful assistant and rigorous mathematician.",
                    user_prompt=user_prompt,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:
                sol = f"Generation error: {exc}"

            solutions[j - 1] = sol if sol else "N/A"
            prev_solution = sol
            time.sleep(sleep_sec)

        result_row: Dict[str, str] = {
            "problem_id": problem_id,
            "parts": str(parts),
            "sub-problem1": sub_probs[0],
            "llm_solution1": solutions[0],
            "sub-problem2": sub_probs[1],
            "llm_solution2": solutions[1],
            "sub-problem3": sub_probs[2],
            "llm_solution3": solutions[2],
            "sub-problem4": sub_probs[3],
            "llm_solution4": solutions[3],
        }

        results.append(result_row)

        if idx % 10 == 0:
            write_rows(output_path, results)

    write_rows(output_path, results)
    # Post-process to correct problem_id if placeholders were used
    fix_output_problem_ids_if_needed(input_path, output_path)


def main(argv: List[str]) -> int:
    input_path = argv[1] if len(argv) > 1 else DEFAULT_INPUT_PATH
    output_path = argv[2] if len(argv) > 2 else DEFAULT_OUTPUT_PATH
    model_name = argv[3] if len(argv) > 3 else DEFAULT_MODEL
    sleep_sec = float(argv[4]) if len(argv) > 4 else DEFAULT_SLEEP_SEC
    max_new_tokens = int(argv[5]) if len(argv) > 5 else DEFAULT_MAX_NEW_TOKENS

    process_all(input_path, output_path, model_name, sleep_sec, max_new_tokens)
    print(f"Done. Solutions saved to: {os.path.abspath(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


