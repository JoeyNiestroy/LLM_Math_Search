"""
This is the core script to collect the last hidden states from the LLM. We run inference all the training data and save to a shard schema. 

THIS WILL GENERATE 2+ TBs of data and TAKE 10+ HOURS on an A100. 

We induce a random order here, to avoid problems down the line with random sampling from a large dataset. For the project it was fixed at random_state = 42

NEW ADDTION 12/9/2024: Added END_EARLY logic for easier testing
"""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np


# Config, batch_size is bascially maxed out. System prefix should not change across code #TODO Probabbly should set gloabl config, file structure is outlined tho
MODEL_ID = "Local_Models/qwen3_3b_local"
INPUT_PATH = "Data/Full_samples.jsonl"
OUT_DIR = Path("qwen3_hidden_states_open_math_sharded")
END_EARLY = False
DTYPE = torch.float16
BATCH_SIZE = 12
MAX_LEN = 4096
SHARD_SIZE = 2000  # number of samples per shard file

SYSTEM_PREFIX = (
    "You are a careful mathematician. Show step-by-step reasoning and label steps like 'Step 1)', 'Step 2)', etc.\n "
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# shard writer class, just accumaltes samples in lists and writes to large .pt files
# Each sample is wrapped in dictonary with metadata, not the most effficent way of saving them, but padding into large matrices is infeasible
#with storage constraints. This allows us to extract and build off this base later, not nessicarly designed to be used a model dataset
class ShardWriter:
    """Accumulates samples and writes them in shards."""
    def __init__(self, out_dir: Path, shard_size: int):
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.current_shard = []
        self.shard_idx = 0
        self.global_idx = 0
        self.catalog = []
        
    def add_sample(self, hidden_states, input_ids, metadata):
        """Add a sample to current shard."""
        sample = {
            'hidden_states': hidden_states.cpu(),  # [L, H]
            'input_ids': input_ids.cpu(),          # [L]
            'metadata': metadata
        }
        self.current_shard.append(sample)
        
        # Write shard if full
        if len(self.current_shard) >= self.shard_size:
            self._write_shard()
    
    def _write_shard(self):
        """Write accumulated samples to a shard file."""
        if not self.current_shard:
            return
            
        shard_path = self.out_dir / f"shard_{self.shard_idx:04d}.pt"
        
        # Save all samples in this shard as a list
        torch.save(self.current_shard, shard_path)
        
        # Update catalog with shard info for each sample
        for local_idx, sample in enumerate(self.current_shard):
            catalog_entry = {
                'id': f"ex_{self.global_idx:06d}",
                'shard_file': f"shard_{self.shard_idx:04d}.pt",
                'shard_idx': self.shard_idx,
                'local_idx': local_idx,
                **sample['metadata']
            }
            self.catalog.append(catalog_entry)
            self.global_idx += 1
        
        print(f"Wrote shard {self.shard_idx} with {len(self.current_shard)} samples")
        
        self.current_shard = []
        self.shard_idx += 1
    
    def finalize(self):
        """Write remaining samples and catalog."""
        # Write final partial shard
        if self.current_shard:
            self._write_shard()
        
        # Write catalog
        catalog_path = self.out_dir / "catalog.jsonl"
        with open(catalog_path, "w", encoding="utf-8") as f:
            for entry in self.catalog:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Wrote catalog with {len(self.catalog)} entries")
        return len(self.catalog)



import random


# Load and shuffle data, confirms schema
records = list(read_jsonl(INPUT_PATH))
assert all(("question" in r and "answer" in r and "is_correct" in r) for r in records), \
    "JSONL must have fields: question, answer, is_correct"

# Shuffle with fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
random.shuffle(records)
print(f"Shuffled {len(records)} records with seed={RANDOM_SEED}")


#prepare output dir
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Load model & tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"
tokenizer.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()


# Main inference loop

shard_writer = ShardWriter(OUT_DIR, SHARD_SIZE)

with torch.no_grad():
    for b_ix, batch in enumerate(chunk(records, BATCH_SIZE)):
        
        
        
        #convert all questions and answers to the expected chat format for the model
        #this is needed to ensure further down the pipeline the hidden states are accurate 
        #to what is seen in inference
        chats = [[
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": str(r["question"])},
            {"role": "assistant", "content": str(r["answer"])},
        ] for r in batch]

        # Render & tokenize
        rendered = tokenizer.apply_chat_template(chats, add_generation_prompt=False, tokenize=False)
        enc = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Forward pass and take last layer
        out = model(**enc, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]  # [B, T, H]

        attn = enc["attention_mask"]
        lengths = attn.sum(dim=1).tolist()
        T = last_hidden.size(1)
        left_pad = (tokenizer.padding_side == "left")

        # Process each sample in batch, I'm sure there's a way to avoid the loop here but it barely impacts perf
        for i, (seq, h, r, L) in enumerate(zip(enc["input_ids"], last_hidden, batch, lengths)):
            L = int(L)
            
            # Trim padding, right padding is allowed but can mess up positional stuff 
            if left_pad:
                h_trim = h[-L:]
                ids_trim = seq[-L:]
            else:
                h_trim = h[:L]
                ids_trim = seq[:L]
            
            # Prepare metadata
            metadata = {
                "question": r["question"],
                "answer": r["answer"],
                "label_correct": int(bool(r["is_correct"])),
                "length": L,
            }
            
            # Add to shard
            shard_writer.add_sample(h_trim, ids_trim, metadata)
        
        if (b_ix + 1) % 10 == 0:
            print(f"Processed {(b_ix + 1) * BATCH_SIZE} samples")
        
        if END_EARLY and b_ix > 10:
            break

# Finalize
total = shard_writer.finalize()
print(f"saved {total} samples to {OUT_DIR.resolve()}")

print(f"Data saved in {shard_writer.shard_idx} shard files")