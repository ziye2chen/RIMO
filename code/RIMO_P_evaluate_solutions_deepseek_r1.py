import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


DEFAULT_REFERENCES_PATH = os.path.join("RIMO", "RIMO-P.csv")
DEFAULT_SOLUTIONS_PATH = os.path.join("RIMO", "RIMO-P_solutions_qwen3_8b_sequential.csv")
DEFAULT_MODEL = "deepseek-r1"
DEFAULT_SLEEP_SEC = 0.25


def _resolve_key(fieldnames: List[str], canonical: str) -> str:
    """Return the actual key name in fieldnames that matches canonical, handling BOM."""
    if canonical in fieldnames:
        return canonical
    bom_key = "\ufeff" + canonical
    if bom_key in fieldnames:
        return bom_key
    for name in fieldnames:
        if name.replace("\ufeff", "") == canonical:
            return name
    return canonical


def load_references(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        key_pid = _resolve_key(fields, "problem_id")
        key_problem = _resolve_key(fields, "problem")
        key_solution = _resolve_key(fields, "solution")
        ref: Dict[str, Dict[str, str]] = {}
        for row in reader:
            pid = (row.get(key_pid) or "").strip()
            if not pid:
                continue
            ref[pid] = {
                "problem": row.get(key_problem, ""),
                "solution": row.get(key_solution, ""),
            }
        return ref


def load_solutions(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # Normalize any BOM on 'problem_id' and 'parts' so downstream code can use canonical names
        fields = reader.fieldnames or []
        key_pid = _resolve_key(fields, "problem_id")
        key_parts = _resolve_key(fields, "parts")
        if key_pid != "problem_id" or key_parts != "parts":
            for r in rows:
                if key_pid in r and "problem_id" not in r:
                    r["problem_id"] = r.get(key_pid, "")
                if key_parts in r and "parts" not in r:
                    r["parts"] = r.get(key_parts, "")
        return rows


def ensure_client() -> OpenAI:
    if not HAS_OPENAI:
        raise ImportError("openai library not available. Install with: pip install openai")
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def build_judge_prompt(problem: str, reference_solution: str, candidate_solution: str, step_index: int, total_parts: int) -> str:
    return (
        "You are an excellent mathematician and a strict judge. Your job is to evaluate the solution of a proof problem. "
        "Only approve if the solution is mathematically correct, logically sound, and free of gaps or unjustified steps.\n\n"
        "Sequential grading protocol (RIMO-P): We grade a proof by the number of consecutive sub-problems solved correctly. "
        f"You are evaluating sub-problem {step_index} of {total_parts}. \n\n"
        "You are given: (1) the original problem; (2) the official complete solution; (3) the candidate sub-solution to evaluate.\n\n"
        "Instructions:\n"
        "- Judge ONLY the candidate sub-solution at this step.\n"
        "- Check correctness, logical validity, and consistency with the problem and official solution.\n"
        "- Be strict: any error, gap, or unjustified claim => incorrect.\n"
        "- Do not grade future steps.\n\n"
        "Respond in STRICT JSON with keys 'verdict' and 'reason'. No extra text.\n"
        "Use one of: {\"verdict\": \"correct\"} or {\"verdict\": \"incorrect\"}.\n"
        "The 'reason' must be a short sentence.\n\n"
        f"Problem:\n{problem}\n\n"
        f"Official complete solution:\n{reference_solution}\n\n"
        f"Candidate sub-solution (step {step_index}):\n{candidate_solution}\n"
    )


def parse_json_response(raw_text: str) -> Tuple[str, str]:
    text = raw_text.strip()
    # Try to extract JSON if wrapped in code fences
    if text.startswith("```"):
        try:
            start = text.index("\n") + 1
            end = text.rindex("```")
            text = text[start:end].strip()
        except ValueError:
            pass
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try substring between first { and last }
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            data = json.loads(text[first : last + 1])
        else:
            raise
    verdict = str(data.get("verdict", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    return verdict, reason


def judge_step(client: OpenAI, model: str, problem: str, reference_solution: str, candidate_solution: str, step_index: int, total_parts: int) -> Tuple[str, str]:
    prompt = build_judge_prompt(problem, reference_solution, candidate_solution, step_index, total_parts)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an excellent mathematician and a strict judge."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = completion.choices[0].message.content or ""
    return parse_json_response(raw)


def evaluate_all(
    references_path: str,
    solutions_path: str,
    model: str,
    sleep_sec: float,
) -> float:
    refs = load_references(references_path)
    rows = load_solutions(solutions_path)
    client = ensure_client()

    # Prepare output CSV next to solutions file
    base = os.path.splitext(os.path.basename(solutions_path))[0]
    out_path = os.path.join(os.path.dirname(solutions_path), f"{base}_judged_{model.replace('/', '_')}.csv")

    fieldnames = [
        "problem_id", "parts", "S_i", "score_i",
        "verdict1", "reason1", "verdict2", "reason2", "verdict3", "reason3", "verdict4", "reason4",
    ]
    # Initialize/overwrite output
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    total_score = 0.0
    num_problems = 0

    for idx, row in enumerate(rows, 1):
        problem_id = (row.get("problem_id") or f"row_{idx:05d}")
        parts_raw = (row.get("parts") or "1").strip()
        try:
            num_parts = max(1, min(4, int(parts_raw)))
        except ValueError:
            num_parts = 1

        ref = refs.get(problem_id)
        if not ref:
            # Skip if no reference available
            continue
        problem = ref.get("problem", "")
        reference_solution = ref.get("solution", "")

        # Collect candidate solutions
        candidate_solutions: List[str] = []
        for j in range(1, 5):
            content = (row.get(f"llm_solution{j}") or "").strip()
            candidate_solutions.append(content if content else "None")

        S_i = 0
        verdicts = ["", "", "", ""]
        reasons = ["", "", "", ""]

        for j in range(1, num_parts + 1):
            cand = candidate_solutions[j - 1]
            if cand.lower() == "none":
                verdicts[j - 1] = "incorrect"
                reasons[j - 1] = "Candidate solution is missing (None)."
                break

            try:
                verdict, reason = judge_step(client, model, problem, reference_solution, cand, j, num_parts)
            except Exception as exc:  # API error -> mark incorrect
                verdict, reason = "incorrect", f"API error: {exc}"

            verdicts[j - 1] = verdict
            reasons[j - 1] = reason
            if verdict == "correct":
                S_i += 1
            else:
                break

            time.sleep(sleep_sec)

        score_i = S_i / float(num_parts)
        total_score += score_i
        num_problems += 1

        # Write incremental result
        with open(out_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "problem_id": problem_id,
                "parts": num_parts,
                "S_i": S_i,
                "score_i": f"{score_i:.6f}",
                "verdict1": verdicts[0],
                "reason1": reasons[0],
                "verdict2": verdicts[1],
                "reason2": reasons[1],
                "verdict3": verdicts[2],
                "reason3": reasons[2],
                "verdict4": verdicts[3],
                "reason4": reasons[3],
            })

    if num_problems == 0:
        print("No problems evaluated (no references matched).")
        return 0.0

    P = total_score / float(num_problems)
    print(f"\nFinal performance score P = {P:.6f} over N={num_problems} problems.")
    print("Judged CSV:", out_path)
    return P


def main(argv: List[str]) -> int:
    references_path = argv[1] if len(argv) > 1 else DEFAULT_REFERENCES_PATH
    solutions_path = argv[2] if len(argv) > 2 else DEFAULT_SOLUTIONS_PATH
    model = argv[3] if len(argv) > 3 else DEFAULT_MODEL
    sleep_sec = float(argv[4]) if len(argv) > 4 else DEFAULT_SLEEP_SEC

    evaluate_all(references_path, solutions_path, model, sleep_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


