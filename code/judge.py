import re, time, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

# ── settings ──────────────────────────────────────────────────────
INPUT_CSV  = "proof_answers_deepseek_math.csv"
OUTPUT_CSV = "proof_scores_deepseek_math.csv"
MODEL_NAME = "Qwen/Qwen3-8B"
MAX_NEW    = 2048
PAUSE_SEC  = 0.5
START_AT   = 1

# ── load model & tokenizer ───────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code= True
)
gen_conf           = GenerationConfig.from_pretrained(MODEL_NAME)
gen_conf.pad_token_id   = model.config.eos_token_id
gen_conf.max_new_tokens = MAX_NEW
model.generation_config = gen_conf
model.eval()

# ── Prepare output files ───────────────────────────────────────
if START_AT == 1:
    # Detailed proof + score
    pd.DataFrame(columns=["problem_id","llm_solution","score"])\
      .to_csv(OUTPUT_CSV, index=False)
    # Scores only
    pd.DataFrame(columns=["problem_id","score"])\
      .to_csv("proof_only_scores_deepseek_math.csv", index=False)


# ── regex to pull out a numeric score ───────────────────────────
score_re = re.compile(r"([-+]?\d+\.?\d*)")

SYSTEM = (
    '''You are a mathematical expert. You need to evaluate the solution of a proof problem. Please follow the steps to give a score to the solution:
Here are 5 key indicators to measure when evaluating a proof solution:
1. Correctness
Logical Rigor: Does the solution logically follow from the given premises and known mathematical principles?
Adherence to Problem Requirements: Does the solution directly address the question and utilize the given conditions appropriately?
Accuracy: Are all mathematical statements, derivations, and conclusions valid?

2. Completeness
Full Proof: Is every step justified, leaving no significant gaps in reasoning?
Handling of Assumptions: Does the solution explicitly consider all assumptions and conditions stated in the problem?

3. Clarity
Structured Presentation: Is the solution organized logically, with clear progression from problem statement to conclusion?
Explanations: Are key steps, methods, and transitions explained clearly and concisely?
Notation Consistency: Is the mathematical notation consistent and aligned with standard practices?

4. Relevance
Appropriateness of Methods: Does the solution use the most relevant and efficient mathematical tools for the problem?
Avoidance of Irrelevant Details: Does the solution focus on solving the problem without introducing unnecessary complications or tangents?

5. Insight
Understanding of Concepts: Does the solution reflect a deep understanding of the underlying mathematical principles and the problem's nuances?
Elegance: If possible, is the solution concise and elegant, avoiding overcomplicated arguments?

Two points for each indicator for a total of 10 points. The solution and the correct solution will be given for your reference. Since the process of the solution is usually wrong, you must be strict with the scores.'''
)

def ask_judge(ref: str, sol: str) -> str:
    # build the messages
    msgs = [
        {"role":"system",  "content": SYSTEM},
        {"role":"user",    "content":
            f"Candidate proof:\n{sol}\n\nReference solution:\n{ref}\n\nScore:"}
    ]
    # this returns a single Tensor of input_ids
    input_ids = tokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    input_ids = input_ids.to(model.device)

    # generate
    with torch.no_grad():
        out = model.generate(
            input_ids       = input_ids,
            max_new_tokens  = gen_conf.max_new_tokens,
            pad_token_id    = gen_conf.pad_token_id,
            eos_token_id    = gen_conf.pad_token_id,
            do_sample       = False,
        )[0]

    # strip off the prompt
    gen_ids = out[input_ids.shape[1] :]
    text    = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    m = score_re.search(text)
    return m.group(1) if m else ""

    # ── load data ────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)


# ── Loop & score ──────────────────────────────────────────────
for idx, row in enumerate(df.itertuples(index=False), start=1):
    if idx < START_AT:
        continue

    pid, ref, sol = row.problem_id, row.reference_solution, row.llm_solution
    print(f"[{idx}/{len(df)}] Judging {pid}…", end=" ", flush=True)

    if isinstance(sol, str) and sol.startswith("ERROR"):
        score = "ERROR"
    else:
        try:
            score = ask_judge(ref, sol)
        except Exception as e:
            score = "ERROR"
            print(f"\n  ▶ Exception on {pid}: {e}")

    # append to proof_scores.csv
    pd.DataFrame(
        [(pid, sol, score)],
        columns=["problem_id","llm_solution","score"]
    ).to_csv(OUTPUT_CSV, mode="a", index=False, header=False)

    # append to proof_only_scores.csv
    pd.DataFrame(
        [(pid, score)],
        columns=["problem_id","score"]
    ).to_csv("proof_only_scores_deepseek_math.csv", mode="a", index=False, header=False)

    print("SCORE:", score, "done.")
    time.sleep(PAUSE_SEC)

print(f"\n✅ All done!\n Detailed results in {OUTPUT_CSV}\n Scores only in proof_only_scores.csv")
