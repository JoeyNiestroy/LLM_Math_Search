"""
This is the main script to generate the data for the eval set. Desgined to just run greedy search on the 'hold out' problem set

While iter_solve works this will work
"""



import json
import torch
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from tqdm import tqdm
from datetime import datetime

# Hard coding can be convienent 
from iter_solve import (
    load_grading_model,
    guided_generation,
    MODEL_ID,
    CHECKPOINT_PATH,
    DTYPE,
)

# Main
def main():
    parser = argparse.ArgumentParser(description="Batch guided generation with grading network")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--k", type=int, default=5, help="Number of candidate completions per iteration")
    parser.add_argument("--max_new", type=int, default=150, help="Max new tokens per iteration")
    parser.add_argument("--threshold", type=float, default=0.95, help="Score threshold to stop")
    parser.add_argument("--max_iters", type=int, default=30, help="Maximum iterations")
    parser.add_argument("--num_problems", type=int, default=1000, help="Number of problems to process")
    parser.add_argument("--output_dir", type=str, default="guided_generation_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print detailed iteration logs")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load problems
    print("Loading problems")
    df = pd.read_csv("Data/Sample_Hard_Problems.csv")
    problems = list(df['Question'])[:args.num_problems]
    print(f"Loaded {len(problems)} problems")

    # Load LLM
    print("Loading language model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, trust_remote_code=True)
    llm.to(device).eval()
    llm_hidden_size = llm.config.hidden_size
    print(f"✓ LLM loaded (hidden_size={llm_hidden_size})")

    # Load grader
    print("Loading grading network...")
    grader = load_grading_model(args.checkpoint, device, llm_hidden_size)
    print("Grading network loaded\n")

    # Process each problem
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{timestamp}.jsonl"
    
    print(f"Processing {len(problems)} problems...")
    print(f"Results will be saved to: {output_file}")
    print("="*80)
    
    for idx, problem in enumerate(tqdm(problems, desc="Processing problems")):
        if args.verbose:
            print(f"\n{'='*80}")
            print(f"Problem {idx+1}/{len(problems)}")
            print(f"{'='*80}")
            print(f"Question: {problem}")
            print(f"{'='*80}")
        
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
        
        result = {
            "problem_idx": idx,
            "problem": problem,
            "final_answer": final_text,
            "final_score": final_score,
        }
        results.append(result)
        
        # Save incrementally
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        if args.verbose:
            print(f"\n{'='*80}")
            print(f"FINAL OUTPUT (Score: {final_score:.4f})")
            print(f"{'='*80}")
            print(final_text)
            print()
    
    # Save summary statistics
    summary_file = output_dir / f"summary_{timestamp}.json"
    summary = {
        "num_problems": len(problems),
        "avg_final_score": sum(r["final_score"] for r in results) / len(results),
        "config": vars(args),
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total problems processed: {len(results)}")
    print(f"Average final score: {summary['avg_final_score']:.4f}")
    print(f"\nResults saved to: {output_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()