import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from FFN_Model import ResidualFFNBinaryClassifier, FFNBinaryClassifier
from pathlib import Path
import argparse

# Config
MODEL_ID = "Local_Models/qwen3_3b_local"
CHECKPOINT_PATH = "checkpoints_ffn/best_model.pt"
SYSTEM_PREFIX = (
    "You are a careful mathematician. Show step-by-step reasoning and label steps like 'Step 1)', 'Step 2)', etc.\n"
)
DTYPE = torch.float32


# Utility: load grading model
def load_grading_model(checkpoint_path, device, llm_hidden_size):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]

    if args.get("use_residual", False):
        model = ResidualFFNBinaryClassifier(
            in_dim=llm_hidden_size,
            hidden_dim=args.get("hidden_dim", 512),
            num_layers=args.get("num_layers", 3),
            dropout=args.get("dropout", 0.1),
            use_layer_norm=args.get("use_layer_norm", False),
        )
    else:
        hidden_dims = [int(d) for d in args.get("hidden_dims", "512,256").split(",")]
        model = FFNBinaryClassifier(
            in_dim=llm_hidden_size,
            hidden_dims=hidden_dims,
            dropout=args.get("dropout", 0.1),
            use_batch_norm=args.get("use_batch_norm", False),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model


# Extract last hidden state for grading
def extract_last_hidden(llm, tokenizer, question, answer, device):
    chat = [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    rendered = tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False , continue_final_message=True)
    enc = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.no_grad():
        out = llm(**enc, output_hidden_states=True, use_cache=False)
                                        #We needs -1 here bc continue_final_message=True
        hidden = out.hidden_states[-1][:, -1, :].squeeze(0).float()
    return hidden


# Iterative generation + grading loop
def guided_generation(llm, tokenizer, grader, device, problem, k=5, max_new=50, threshold=0.95, max_iters=10, verbose = True):
    current_text = ""
    
    for iteration in range(max_iters):
        # Build the chat with current progress
        chat = [
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": current_text},
        ]
                                                                                                    #This is important, needs to be True
        rendered = tokenizer.apply_chat_template(chat, add_generation_prompt=False, tokenize=False , continue_final_message=True)
        enc = tokenizer(rendered, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        
        prompt_length = enc['input_ids'].shape[-1]

        # Sample K completions
        with torch.no_grad():
            outputs = llm.generate(
                **enc,
                do_sample=True,
                top_p=0.9,
                temperature=1.1,
                max_new_tokens=max_new,
                num_return_sequences=k,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        candidates = []
        eos_found = False
        
        for i in range(k):
            # Extract only the newly generated tokens
            generated_ids = outputs[i][prompt_length:]
            
            # Check if EOS was generated in the NEW tokens
            if tokenizer.eos_token_id in generated_ids:
                eos_found = True
            
            gen_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            full_answer = current_text + gen_text

            # Grade the new completion
            hidden = extract_last_hidden(llm, tokenizer, problem, full_answer, device)
            with torch.no_grad():
                score = grader(hidden.unsqueeze(0))["probs"].item()
            candidates.append((score, gen_text, full_answer))

        # Pick best candidate
        best_score, best_gen, best_answer = max(candidates, key=lambda x: x[0])
        
        if verbose:
            print(f"\nIteration {iteration+1}: best_score={best_score:.4f}")
            print("------------")
            print(best_gen.strip())
            print()

        current_text = best_answer

        # Stopping conditions
        if best_score > threshold:
            if verbose:
                print(f"Stopping: Score threshold {threshold} reached (score={best_score:.4f})")
            break
        
        if eos_found:
            if verbose:
                print("Stopping: EOS token generated")
            break
        
        # Check if no new text was generated
        if not best_gen.strip():
            if verbose:
                print("Stopping: No new text generated")
            break

    return current_text, best_score


# Main
def main():
    parser = argparse.ArgumentParser(description="Guided generation with grading network")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--k", type=int, default=5, help="Number of candidate completions per iteration")
    parser.add_argument("--max_new", type=int, default=150, help="Max new tokens per iteration")
    parser.add_argument("--threshold", type=float, default=0.95, help="Score threshold to stop")
    parser.add_argument("--max_iters", type=int, default=30, help="Maximum iterations")
    parser.add_argument("--problem", type=str, default=None, help="Math problem to solve")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load LLM
    print("Loading language model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, trust_remote_code=True)
    llm.to(device).eval()
    llm_hidden_size = llm.config.hidden_size
    print(f"LLM loaded (hidden_size={llm_hidden_size})")

    # Load grader
    print("Loading grading network...")
    grader = load_grading_model(args.checkpoint, device, llm_hidden_size)
    print("Grading network loaded\n")

    # Define problem
    problem = args.problem if args.problem else "Determine all positive integers n for which there exist positive integers a, b, and c satisfying $2a^n +3b^n = 4c^n$"

    print("="*80)
    print(f"Problem: {problem}")
    print("="*80)

    final_text, final_score = guided_generation(
        llm,
        tokenizer,
        grader,
        device,
        problem,
        k=args.k,
        max_new=args.max_new,
        threshold=args.threshold,
        max_iters=args.max_iters,
    )

    print("\n" + "="*80)
    print("FINAL OUTPUT")
    print("="*80)
    print(final_text)
    print(f"\nFinal grading probability: {final_score:.4f}")


if __name__ == "__main__":
    main()
