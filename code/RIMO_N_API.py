"""
Evaluate IMO_try.csv with DeepSeek-R1-671B via DashScope OpenAI‐compatible API,
saving each answer as soon as it’s produced.
Prerequisites
  pip install openai pandas
"""
import os
import re
import time
import pandas as pd
from openai import OpenAI

# ── 1. API setup ──────────────────────────────────────────────
client = OpenAI(
    api_key="your_api_key_here",
    base_url="your_base_url_here",
)

MODEL_NAME = "qwq-32b"  # or replace with your preferred model
PROMPT_TAIL = (
    "\n\nPlease think step by step, and put your final answer within \\boxed{}.\n\n"
)

boxed_re = re.compile(r"\\boxed\{([^}]*)\}")
def extract_boxed(text: str) -> str | None:
    m = boxed_re.findall(text)
    return m[-1].strip() if m else None

def ask_gpt(question: str) -> str:
    parts = []
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert mathematician. Show detailed reasoning."},
            {"role": "user",   "content": question + PROMPT_TAIL},
        ],
        stream=True,
        max_tokens=2048,
    )
    for chunk in completion:
        choice = chunk.choices[0]
        txt    = getattr(choice.delta, "content", None)
        if txt:
            parts.append(txt)
        if choice.finish_reason is not None:
            break
    return "".join(parts)

# ── 2. load dataset ───────────────────────────────────────────
CSV_IN  = "../RIMO/RIMO-N.csv"
CSV_OUT = "answer_qwq.csv"

df = pd.read_csv(CSV_IN)[["problem_id", "problem", "answer"]]
# df = df.rename(columns={"answer": "correct_answer"})

# # ── 3. prepare output file ────────────────────────────────────
# # write header only once
# pd.DataFrame(columns=["problem_id", "correct_answer", "llm_answer"])\
#   .to_csv(CSV_OUT, index=False)

import pandas as pd
import time
import re
from openai import OpenAI

# … your client, ask_gpt, extract_boxed, loading df, CSV_OUT, etc …

# load what’s already done
done_df       = pd.read_csv(CSV_OUT)
completed_ids = set(done_df["problem_id"].astype(str))

# filter to the ones we still need
to_do = df[~df["problem_id"].astype(str).isin(completed_ids)].copy()

print(f"{len(completed_ids)} problems already done, {len(to_do)} to go…")

MAX_TOTAL = 400

for idx, (pid, statement, gold) in enumerate(
        to_do.itertuples(index=False),
        start=1 + len(completed_ids)
    ):
    # stop once we hit the 50th overall
    if idx > MAX_TOTAL:
        print(f"Reached problem {MAX_TOTAL}; stopping.")
        break

    print(f"Solving Problem {idx} (ID={pid}) …", end=" ")
    try:
        reply = ask_gpt(statement)
        time.sleep(1.0)
        pred  = extract_boxed(reply) or ""
    except Exception as err:
        pred = f"ERROR: {err}"

    # append this result
    pd.DataFrame(
        [(pid, gold, pred)],
        columns=["problem_id", "correct_answer", "llm_answer"]
    ).to_csv(CSV_OUT, mode="a", header=False, index=False)

    print("✅")

print("Done.")
