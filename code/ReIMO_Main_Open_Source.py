import re, time, pandas as pd, torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 0. constants ───────────────────────────────────────────────
CSV_PATH    = "ReIMO.csv"
OUTPUT_CSV  = "proof_answer_qwen3_8b.csv"
TAIL = (
    "\n\nPlease think step by step, and put your final answer within "
    "\\boxed{}.\n\n"
)
boxed_re = re.compile(r"\\boxed\{([^}]*)\}")

def extract_boxed(text: str):
    m = boxed_re.findall(text)
    return m[-1].strip() if m else ""

# ── 1. load model & tokenizer ───────────────────────────────────
model_name = "Qwen/Qwen3-8B"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()



# ── 2. load dataset ─────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.rename(columns={
    "problem_id":   "problem_id",
    "problem":      "problem",
    "answer":       "correct_answer",
})[["problem_id", "problem", "correct_answer"]]

# ── 3. prepare output file ──────────────────────────────────────
# Remove old file if exists
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)
# Write header
pd.DataFrame(columns=["problem_id", "correct_answer", "llm_answer"])\
  .to_csv(OUTPUT_CSV, index=False)

# ── 4. inference loop (append each result) ──────────────────────
start = time.time()
for idx, row in df.iterrows():
    pid, stmt, gold = row
    print(f"[{idx+1}/{len(df)}] Solving id={pid}…", end=" ", flush=True)

    # build input
    messages = [{"role": "user", "content": stmt + TAIL}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # generate
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=8192
        )[0].tolist()

    # strip prompt tokens
    prompt_len = inputs.input_ids.shape[1]
    output_ids = gen_ids[prompt_len:]

    # split off reasoning
    try:
        split_idx = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        split_idx = 0

    content_ids = output_ids[split_idx:]
    content = tokenizer.decode(content_ids, skip_special_tokens=True).strip()
    pred    = extract_boxed(content)

    # append to CSV immediately
    pd.DataFrame(
        [(pid, gold, pred)],
        columns=["problem_id", "correct_answer", "llm_answer"]
    ).to_csv(OUTPUT_CSV, mode="a", index=False, header=False)

    print("done.")

end = time.time()
print(f"✅ Finished in {(end-start)/60:.1f} min. Results in {OUTPUT_CSV}")
