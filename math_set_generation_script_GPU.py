# save as build_hidden_states_generation_sharded.py
import os, json, math
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Inference_Shard import ShardWriter


# Config
MODEL_ID = "./Local_Models/qwen3_3b_local"

DTYPE = torch.float16
BATCH_SIZE = 12
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.05

SHARD_SIZE = 2000
OUT_DIR = Path("qwen3_hidden_states_sample_problems")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("Data/Sample_Hard_Problems.csv")
MATH_PROBLEMS = list(df['Question'])[0:1000]

SYSTEM_PREFIX = (
    "You are a careful mathematician. Show step-by-step reasoning and label steps like 'Step 1)', 'Step 2)', etc.\n"
)



def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]



# Load model & tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"


try:
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id >= 0:
        EOS_TOKEN = im_end_id
    else:
        EOS_TOKEN = tokenizer.eos_token_id
except:
    EOS_TOKEN = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    trust_remote_code=True,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()


# Main Loop
shard_writer = ShardWriter(OUT_DIR, SHARD_SIZE)

with torch.no_grad():
    for batch_ix, batch_prompts in enumerate(chunks(MATH_PROBLEMS, BATCH_SIZE)):

        # Build chats
        chat_batches = []
        for p in batch_prompts:
            chat_batches.append([
                {"role": "system", "content": SYSTEM_PREFIX},
                {"role": "user",   "content": p},
            ])

        # Render to chat template and tokenize
        rendered = tokenizer.apply_chat_template(
            chat_batches,
            add_generation_prompt=True,
            tokenize=False,
        )

        enc = tokenizer(rendered, return_tensors="pt", padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        print("Input Ids Shape:", tuple(enc["input_ids"].shape))


        # GENERATION FOR THE ANSWER
        gen_out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=REPETITION_PENALTY,
            do_sample=True,
            eos_token_id=EOS_TOKEN,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
        )
        sequences = gen_out.sequences  # [B, T_full]

        # Decode output texts
        texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)


        # SECOND FORWARD: HIDDEN STATES
        # Yes we could use output hidden states above but I have disagreements with the hugging face team, I like code to make sense they don't
        attn_mask = (sequences != tokenizer.pad_token_id).long().to(device)

        out = model(
            input_ids=sequences.to(device),
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = out.hidden_states[-1]   # [B, T, H]

        lengths = attn_mask.sum(dim=1).tolist()
        T = last_hidden.size(1)
        left_pad = tokenizer.padding_side == "left"


        # PROCESS EACH SAMPLE
        for i, (problem, text, seq, h, L) in enumerate(zip(batch_prompts, texts, sequences, last_hidden, lengths)):
            L = int(L)

            if left_pad:
                h_trim = h[-L:]
                ids_trim = seq[-L:]
            else:
                h_trim = h[:L]
                ids_trim = seq[:L]

            metadata = {
                "problem": problem,
                "answer_text": text,
                "length": L,
                "padded_to": int(T),
            }

            shard_writer.add_sample(h_trim, ids_trim, metadata)

        print(f" Completed batch {batch_ix+1} ({len(batch_prompts)} examples)")

# Finalize
total = shard_writer.finalize()
print(f"\n Saved {total} samples into {shard_writer.shard_idx} shard files at {OUT_DIR.resolve()}")
