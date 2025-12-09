# save as build_hidden_states_from_jsonl.py
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config (edit as needed)
# -------------------------
MODEL_ID   = "/scratch/nyr5zq/Local_Models/qwen3_3b_local"
INPUT_PATH = "Data/OpenMathInstruct_samples.jsonl"   # JSONL with fields: question, answer, is_correct
OUT_DIR    = Path("qwen3_hidden_states_open_math_1")

DTYPE  = torch.float16
BATCH_SIZE = 12
MAX_LEN    = 4096

SYSTEM_PREFIX = (
    "You are a careful mathematician. Show step-by-step reasoning and label steps like 'Step 1)', 'Step 2)', etc.\n "
)

# -------------------------
# IO helpers
# -------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

# -------------------------
# Load data
# -------------------------
records = list(read_jsonl(INPUT_PATH))
assert all(("question" in r and "answer" in r and "is_correct" in r) for r in records), \
    "JSONL must have fields: question, answer, is_correct"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load model & tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side   = "left"
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# -------------------------
# Main
# -------------------------
catalog_path = OUT_DIR / "catalog.jsonl"
total = 0


with torch.no_grad(), open(catalog_path, "w", encoding="utf-8") as cat:
    for b_ix, batch in enumerate(chunk(records, BATCH_SIZE)):
        chats = [[
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": str(r["question"])},
            {"role": "assistant", "content": str(r["answer"])},
        ] for r in batch]

        # render & tokenize
        rendered = tokenizer.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # forward
        out = model(**enc, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]  # [B, T, H]

        attn = enc["attention_mask"]         # [B, T] with 1 for real tokens
        lengths = attn.sum(dim=1).tolist()   # per-sample true lengths
        T = last_hidden.size(1)
        left_pad = (tokenizer.padding_side == "left")

        # save tensors + metadata (TRIMMED to true length)
        for i, (seq, h, r, L) in enumerate(zip(enc["input_ids"], last_hidden, batch, lengths)):
            L = int(L)
            if left_pad:
                # real tokens are the *rightmost* L positions
                h_trim   = h[-L:].to("cpu")          # [L, H]
                ids_trim = seq[-L:].to("cpu")        # [L]
                start_ix = T - L                     # where content starts in padded tensor
            else:
                # real tokens are the *leftmost* L positions
                h_trim   = h[:L].to("cpu")
                ids_trim = seq[:L].to("cpu")
                start_ix = 0

            item_id = f"ex_{b_ix:06d}_{i:02d}"

            torch.save(h_trim,   OUT_DIR / f"{item_id}_hidden.pt")
            torch.save(ids_trim,        OUT_DIR / f"{item_id}_input_ids.pt")

            meta = {
                "id": item_id,
                "question": r["question"],
                "answer": r["answer"],
                "label_correct": int(bool(r["is_correct"])),

                # bookkeeping
                "length": L,                    # true tokens saved
                "padded_to": int(T),            # original padded length
                "start_index_in_padded": start_ix,
            }
            cat.write(json.dumps(meta, ensure_ascii=False) + "\n")
            total += 1

        print(f"✅ Batch {b_ix + 1} complete ({len(batch)} examples)")

print(f"All done — saved {total} samples to {OUT_DIR.resolve()}")
