import os
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── 0. Paths & settings ───────────────────────────────────────────
CSV_IN       = "../RIMO/RIMO-P.csv"                # input with cols: problem_id, problem, solution
LLM_OUT      = "proof_llm_solutions.csv"      # incremental LLM‐only output
CSV_OUT      = "proof_answers_qwen3_8b.csv"   # final merged output
MODEL        = "Qwen/Qwen3-8B"
SLEEP_SEC    = 0.5                             # pause between calls
MAX_NEW      = 512                             # length of generated proof

# ── 1. Load model & tokenizer ────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
pad_token_id = model.config.eos_token_id       # use EOS as pad

# ── 2. Prepare incremental LLM output CSV ─────────────────────────
if os.path.exists(LLM_OUT):
    os.remove(LLM_OUT)
pd.DataFrame(columns=["problem_id", "llm_solution"])\
  .to_csv(LLM_OUT, index=False)

# ── 3. Load just the problems ─────────────────────────────────────
df_problems = pd.read_csv(CSV_IN)[["problem_id", "problem"]]
df_problems = df_problems.rename(columns={"problem": "problem_text"})

# ── 4. Generate proofs & save each immediately ───────────────────
for idx, row in enumerate(df_problems.itertuples(index=False), start=1):
    pid, text = row.problem_id, row.problem_text
    print(f"[{idx}/{len(df_problems)}] Generating proof for ID={pid} …", end=" ", flush=True)

    # Build chat‐style prompt
    user_msg = {
        "role":    "user",
        "content": (
            "Please write a complete, step-by-step proof of the following problem:\n\n"
            f"{text}\n\n"
            "End with a brief concluding statement."
        )
    }

    # Tokenize via apply_chat_template
    batch = tokenizer.apply_chat_template(
        [user_msg],
        add_generation_prompt=True,
        return_tensors="pt"
    )
    # Handle both dict-and-Tensor returns
    if isinstance(batch, torch.Tensor):
        input_ids = batch.to(model.device)
        attention_mask = None
    else:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        input_ids      = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

    # Generate
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
            do_sample=False,
        )

    # Strip prompt and decode
    seq        = out[0]
    prompt_len = input_ids.shape[1]
    proof_ids  = seq[prompt_len:].tolist()
    llm_proof  = tokenizer.decode(proof_ids, skip_special_tokens=True).strip()

    # Append this single result to LLM_OUT
    pd.DataFrame(
        [(pid, llm_proof)],
        columns=["problem_id", "llm_solution"]
    ).to_csv(LLM_OUT, mode="a", index=False, header=False)

    print("done.")
    time.sleep(SLEEP_SEC)

# ── 5. Merge LLM outputs with reference solutions ────────────────
df_llm = pd.read_csv(LLM_OUT)
df_refs = pd.read_csv(CSV_IN)[["problem_id", "solution"]]
df_refs = df_refs.rename(columns={"solution": "reference_solution"})

df_out = df_refs.merge(df_llm, on="problem_id")\
                [["problem_id", "reference_solution", "llm_solution"]]

# ── 6. Write final CSV ───────────────────────────────────────────
df_out.to_csv(CSV_OUT, index=False)
print(f"\n✅ All proofs generated and merged! See: {CSV_OUT}")
