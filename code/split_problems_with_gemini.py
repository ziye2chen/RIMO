import csv
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    # Gemini official Python SDK (google-genai)
    from google import genai
except ImportError as exc:
    raise SystemExit(
        "Missing dependency 'google-genai'. Install with: pip install google-genai"
    ) from exc


DEFAULT_INPUT_PATH = os.path.join("RIMO", "RIMO-P_with_parts.csv")
DEFAULT_OUTPUT_PATH = os.path.join("RIMO", "RIMO-P_parts_expanded.csv")
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")


def read_rows(input_csv_path: str) -> List[Dict[str, str]]:
    with open(input_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"problem_id", "problem", "solution", "solution_word_count", "parts"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Input CSV missing required columns: {sorted(missing)}")
        rows: List[Dict[str, str]] = []
        for row in reader:
            rows.append(row)
        return rows


def write_header(writer: csv.writer) -> None:
    header = [
        "problem_id",
        "problem",
        "solution",
        "number_of_parts",
        "sub-problem1",
        "sub-solution1",
        "sub-problem2",
        "sub-solution2",
        "sub-problem3",
        "sub-solution3",
        "sub-problem4",
        "sub-solution4",
    ]
    writer.writerow(header)


def ensure_client() -> genai.Client:
    # The SDK can read GOOGLE_API_KEY from environment implicitly, but we allow explicit pass-through
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # The client can still initialize without explicit key if configured externally,
        # but it's better to guide the user early.
        sys.stderr.write(
            "Warning: GOOGLE_API_KEY not set. Set it to your Gemini API key.\n"
        )
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    return client


def build_prompt(problem: str, solution: str, target_parts: int) -> str:
    capped_parts = max(1, min(4, target_parts))
    # Instruction ensures the final sub-problem matches the original problem's final goal
    return f"""
You are a rigorous math assistant helping to split a proof problem into staged sub-problems.
Given the original problem statement and a complete solution, split the proof into exactly {capped_parts} logically progressive sub-problems and their sub-solutions. The last sub-problem must be the final goal from the original problem. Earlier sub-problems should be prerequisite lemmas/theorems that build toward the final proof.

Output STRICT JSON with this shape (no extra commentary):
{{
  "number_of_parts": <int>,
  "parts": [
    {{
      "sub_problem": <string>,
      "sub_solution": <string>
    }}
  ]
}}

Constraints:
- Produce exactly {capped_parts} items in 'parts'.
- The final 'sub_problem' is the original problem's final statement/goal.
- Each sub_solution must be concise and self-contained.
- Do not include markdown or explanations outside the JSON.

Original problem:
{problem}

Complete solution:
{solution}
"""


def call_gemini_with_retry(client: genai.Client, prompt: str, model: str) -> str:
    max_retries = 5
    base_delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if not text:
                raise RuntimeError("Empty response text from Gemini")
            return text.strip()
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                raise
            sleep_s = base_delay * (2 ** attempt) + 0.25 * (attempt + 1)
            sys.stderr.write(f"Gemini call failed (attempt {attempt+1}/{max_retries}): {exc}\n")
            time.sleep(sleep_s)
    raise RuntimeError("Unreachable")


def _sanitize_json_backslashes(text: str) -> str:
    r"""Escape invalid backslashes inside JSON string values.

    Strategy:
    - Double every backslash that is not already part of a valid JSON escape sequence.
    - Valid JSON escapes: \" \\ \/ \b \f \n \r \t \\uXXXX
    """
    import re

    # First, protect valid escapes by temporarily marking them
    protected = {
        '\\\\"': '__ESC_DQ__',
        '\\\\\\\\\\': '__ESC_BS__',
        '\\\\/': '__ESC_SLASH__',
        '\\\\b': '__ESC_b__',
        '\\\\f': '__ESC_f__',
        '\\\\n': '__ESC_n__',
        '\\\\r': '__ESC_r__',
        '\\\\t': '__ESC_t__',
    }

    def protect(s: str) -> str:
        for k, v in protected.items():
            s = s.replace(k, v)
        # Protect \uXXXX sequences
        s = re.sub(r"\\\\u([0-9a-fA-F]{4})", r"__ESC_u_\1__", s)
        return s

    def unprotect(s: str) -> str:
        s = re.sub(r"__ESC_u_([0-9a-fA-F]{4})__", r"\\u\1", s)
        for k, v in protected.items():
            s = s.replace(v, k)
        return s

    tmp = protect(text)
    # Any remaining single backslash should be doubled
    tmp = tmp.replace('\\', '\\\\')
    repaired = unprotect(tmp)
    return repaired


def parse_parts_json(raw_text: str, fallback_parts: int) -> Tuple[int, List[Tuple[str, str]]]:
    # Some models may wrap JSON in code fences; try to extract JSON block if present.
    text = raw_text.strip()
    if text.startswith("```") or text.startswith("```json"):
        # crude extraction
        try:
            start = text.index("\n") + 1
            end = text.rindex("```")
            text = text[start:end].strip()
        except ValueError:
            pass

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Attempt to find a JSON object inside the text
        first = text.find("{")
        last = text.rfind("}")
        candidate = text[first : last + 1] if (first != -1 and last != -1 and last > first) else text
        # Try direct candidate
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            # As a fallback, try sanitizing invalid backslashes
            repaired = _sanitize_json_backslashes(candidate)
            data = json.loads(repaired)

    number_of_parts = int(data.get("number_of_parts", fallback_parts))
    parts_list = data.get("parts", [])
    extracted: List[Tuple[str, str]] = []
    for item in parts_list:
        sub_problem = (item.get("sub_problem") or "").strip()
        sub_solution = (item.get("sub_solution") or "").strip()
        if sub_problem or sub_solution:
            extracted.append((sub_problem, sub_solution))
    return number_of_parts, extracted


def process_row(
    client: genai.Client,
    row: Dict[str, str],
    model: str,
) -> Tuple[str, str, str, int, List[Optional[str]]]:
    problem_id = row.get("problem_id", "").strip()
    problem = row.get("problem", "").strip()
    solution = row.get("solution", "").strip()
    parts_raw = (row.get("parts") or "1").strip()

    try:
        requested_parts = int(parts_raw)
    except ValueError:
        requested_parts = 1

    # We only have columns for up to 4 parts; cap here for consistency.
    target_parts = max(1, min(4, requested_parts))

    if target_parts == 1:
        subs: List[Optional[str]] = [problem, solution]
        # pad remaining 3 pairs with literal "None"
        for _ in range(3):
            subs.extend(["None", "None"])
        return problem_id, problem, solution, 1, subs

    prompt = build_prompt(problem=problem, solution=solution, target_parts=target_parts)
    raw = call_gemini_with_retry(client, prompt, model=model)
    gen_parts_count, parts_list = parse_parts_json(raw, fallback_parts=target_parts)

    # Normalize to exactly target_parts, prefer model's count but still cap to 4
    normalized_parts = max(1, min(4, gen_parts_count))
    if normalized_parts != target_parts:
        # If mismatch, truncate or pad as needed
        target_parts = normalized_parts

    # Ensure exactly target_parts entries
    parts_list = parts_list[:target_parts]
    while len(parts_list) < target_parts:
        parts_list.append(("", ""))

    # Flatten into [sub-problem1, sub-solution1, ..., sub-problem4, sub-solution4]
    flattened: List[Optional[str]] = []
    for sub_problem, sub_solution in parts_list:
        flattened.append(sub_problem if sub_problem else "None")
        flattened.append(sub_solution if sub_solution else "None")
    # pad up to 4 pairs
    for _ in range(4 - target_parts):
        flattened.extend(["None", "None"])

    return problem_id, problem, solution, target_parts, flattened


def main(argv: List[str]) -> int:
    input_path = argv[1] if len(argv) > 1 else DEFAULT_INPUT_PATH
    output_path = argv[2] if len(argv) > 2 else DEFAULT_OUTPUT_PATH
    model = argv[3] if len(argv) > 3 else DEFAULT_MODEL

    rows = read_rows(input_path)
    client = ensure_client()

    with open(output_path, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        write_header(writer)

        for idx, row in enumerate(rows, start=1):
            try:
                problem_id, problem, solution, num_parts, flattened = process_row(
                    client, row, model
                )
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(
                    f"Row {idx} (problem_id={row.get('problem_id')}): failed with {exc}. Filling Nones.\n"
                )
                problem_id = row.get("problem_id", "")
                problem = row.get("problem", "")
                solution = row.get("solution", "")
                num_parts = 1
                flattened = [problem, solution]
                for _ in range(3):
                    flattened.extend([None, None])

            writer.writerow([
                problem_id,
                problem,
                solution,
                num_parts,
                *flattened,
            ])

    print(
        f"Wrote expanded parts CSV with Gemini splits to: {os.path.abspath(output_path)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


