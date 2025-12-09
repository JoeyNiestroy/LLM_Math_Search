"""
This contains the code for MCTS algo, modeled after alphago/alphazero. Priors are swapped with normalized log odds of the sequences. 
VERY SLOW even on GPU. Rollouts are expensive, need a cheaper action network or better desgined code. 

"""



import math
import torch
import torch.nn.functional as F
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
DTYPE = torch.float16



# Utility: load grading model
def load_grading_model(checkpoint_path, device, llm_hidden_size):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = ckpt["args"]


    # I'll leave this here but don't use this old model
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
    """
    Build chat: [system, user=question, assistant=answer]
    and return last hidden state vector (shape: [hidden_dim]).
    """
    chat = [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    rendered = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=False,
        tokenize=False,
        continue_final_message=True,
    )
    enc = tokenizer(rendered, return_tensors="pt").to(device)
    with torch.no_grad():
        out = llm(**enc, output_hidden_states=True, use_cache=False)
        # last layer, last token, shape: [hidden_dim]
        hidden = out.hidden_states[-1][:, -1, :].squeeze(0).float()
    return hidden


def grade_prefix(llm, tokenizer, grader, device, problem, answer_prefix):
    """
    Grade the current prefix (partial solution).
    Returns a scalar probability (float) that this prefix leads to a correct final answer.
    """
    hidden = extract_last_hidden(llm, tokenizer, problem, answer_prefix, device)
    with torch.no_grad():
        prob = grader(hidden.unsqueeze(0))["probs"].item()
    return prob




# MCTS components

class TreeNode:
    """
    Node in the MCTS tree.

    Each node corresponds to a *prefix* of the assistant's answer: current_text.
    Edges/actions correspond to appending a new chunk of text (gen_text_chunk).
    """

    def __init__(self, parent, prior, current_text, depth=0, eos_reached=False):
        self.parent = parent         # TreeNode or None
        self.prior = prior           # P(s,a) from move network (LLM)
        self.current_text = current_text
        self.depth = depth
        self.eos_reached = eos_reached

        self.children = []           # list of (action_text, child_node)
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_terminal = False     # set when we decide not to expand further

    @property
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def is_expanded(self):
        return len(self.children) > 0

    def add_child(self, action_text, child_node):
        self.children.append((action_text, child_node))


def puct_score(parent, child_node, child_prior, c_puct=1.0):
    """
    PUCT score used during selection:
    UCB = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
    """
    if child_node.visit_count > 0:
        q = child_node.value
    else:
        q = 0.0
    n_parent = max(1, parent.visit_count)
    u = c_puct * child_prior * math.sqrt(n_parent) / (1 + child_node.visit_count)
    return q + u


def select_child(node, c_puct=1.0):
    """
    Select child with maximum PUCT score.
    node.children is a list of (action_text, child_node).
    """
    best_score = -float("inf")
    best_child = None
    best_action = None

    for action_text, child in node.children:
        score = puct_score(node, child, child.prior, c_puct=c_puct)
        if score > best_score:
            best_score = score
            best_child = child
            best_action = action_text

    return best_action, best_child


def propose_candidates_with_priors(
    llm,
    tokenizer,
    device,
    problem,
    current_text,
    k=5,
    max_new=50,
    top_p=0.9,
    temperature=1.1,
):
    """
    From a given prefix (current_text), generate K candidate chunks and compute
    their sequence log-probabilities under the LLM.

    Returns:
        candidates: list of dict with keys:
            'chunk'      : generated chunk text (string)
            'new_text'   : full new prefix (current_text + chunk)
            'logprob'    : scalar log prob of the generated tokens
            'eos'        : whether EOS was generated in the new tokens
    """
    chat = [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": current_text},
    ]
    rendered = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=False,
        tokenize=False,
        continue_final_message=True,
    )
    enc = tokenizer(rendered, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    prompt_length = enc["input_ids"].shape[-1]

    with torch.no_grad():
        gen_outputs = llm.generate(
            **enc,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

    sequences = gen_outputs.sequences  # [k, total_len]
    scores = gen_outputs.scores        # list length = #new_tokens, each [k, vocab]

    candidates = []

    # sequences shape: [batch=k, total_len]
    # we only care about tokens after prompt_length
    for i in range(k):
        generated_ids = sequences[i, prompt_length:]
        eos_found = False

        # Compute sequence log-prob under the model
        # scores[t] is logits for the t-th generated token (before softmax),
        # shape: [k, vocab]. For sample i, we take scores[t][i]
        logprob = 0.0
        # Note: length of scores list equals max length of generated steps
        # Some sequences may have ended early, but we can still compute
        # log-probs until EOS and then stop.
        for t, token_id in enumerate(generated_ids):
            # If we hit EOS or PAD, we include it and break.
            logits_t = scores[t][i]  # [vocab]
            log_probs_t = F.log_softmax(logits_t, dim=-1)
            logprob += log_probs_t[token_id].item()

            if token_id.item() == tokenizer.eos_token_id:
                eos_found = True
                break

        gen_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
        new_text = current_text + gen_text

        candidates.append(
            {
                "chunk": gen_text,
                "new_text": new_text,
                "logprob": logprob,
                "eos": eos_found,
            }
        )

    # Turn logprobs into normalized priors via softmax
    logprobs_tensor = torch.tensor([c["logprob"] for c in candidates], dtype=torch.float32)
    priors = F.softmax(logprobs_tensor, dim=0).tolist()
    for c, p in zip(candidates, priors):
        c["prior"] = float(p)

    return candidates


def mcts_search_once(
    llm,
    tokenizer,
    grader,
    device,
    problem,
    root,
    num_simulations=32,
    c_puct=1.0,
    max_depth=3,
):
    """
    Run MCTS starting from a given root node (prefix = root.current_text).

    After search, root.children has candidate moves with visit counts and values.
    We do *not* modify root.current_text itself.
    """
    for _ in range(num_simulations):
        node = root
        path = [node]

        # 1) SELECTION:
        # Traverse down the tree using PUCT until we reach a leaf (unexpanded) or terminal node.
        while node.is_expanded() and (not node.is_terminal) and (not node.eos_reached):
            action_text, next_node = select_child(node, c_puct=c_puct)
            node = next_node
            path.append(node)

            if node.depth >= max_depth:
                break

        # 2) EVALUATION & EXPANSION:
        # Evaluate the current node's prefix with the grading network.
        value = grade_prefix(llm, tokenizer, grader, device, problem, node.current_text)

        # Terminal condition check:
        if node.depth >= max_depth or node.eos_reached:
            node.is_terminal = True
        else:
            # Expand this leaf node if not already expanded
            if not node.is_expanded():
                candidates = propose_candidates_with_priors(
                    llm=llm,
                    tokenizer=tokenizer,
                    device=device,
                    problem=problem,
                    current_text=node.current_text,
                    k=5,          # you can make this a parameter
                    max_new=50,   # or pass through args
                )

                if len(candidates) == 0:
                    node.is_terminal = True
                else:
                    for cand in candidates:
                        child = TreeNode(
                            parent=node,
                            prior=cand["prior"],
                            current_text=cand["new_text"],
                            depth=node.depth + 1,
                            eos_reached=cand["eos"],
                        )
                        node.add_child(cand["chunk"], child)

        # 3) BACKUP:
        # Propagate the value back along the path.
        for n in path:
            n.visit_count += 1
            n.value_sum += value


def mcts_guided_generation(
    llm,
    tokenizer,
    grader,
    device,
    problem,
    k_root=5,
    max_new=50,
    threshold=0.95,
    max_iters=10,
    num_simulations=64,
    c_puct=1.0,
    max_depth=3,
):
    """
    High-level loop similar to guided_generation, but at each iteration:
      - Root is the current prefix
      - We expand via MCTS
      - Choose the child of root with highest visit count (or value) as the next chunk
    """
    current_text = ""

    for iteration in range(max_iters):
        print(f"\n====== MCTS Iteration {iteration+1} ======")

        # Create root node for this iteration with current prefix
        root = TreeNode(parent=None, prior=1.0, current_text=current_text, depth=0)

        # First expand root once to get initial children (K candidates)
        root_candidates = propose_candidates_with_priors(
            llm=llm,
            tokenizer=tokenizer,
            device=device,
            problem=problem,
            current_text=current_text,
            k=k_root,
            max_new=max_new,
        )

        if len(root_candidates) == 0:
            print("No candidates from root; stopping.")
            break

        for cand in root_candidates:
            child = TreeNode(
                parent=root,
                prior=cand["prior"],
                current_text=cand["new_text"],
                depth=1,
                eos_reached=cand["eos"],
            )
            root.add_child(cand["chunk"], child)

        # Now run MCTS simulations starting from this root
        mcts_search_once(
            llm=llm,
            tokenizer=tokenizer,
            grader=grader,
            device=device,
            problem=problem,
            root=root,
            num_simulations=num_simulations,
            c_puct=c_puct,
            max_depth=max_depth,
        )

        # Choose the child of root with the highest visit count as the next move
        best_child = None
        best_action = None
        best_visits = -1

        for action_text, child in root.children:
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_child = child
                best_action = action_text

        if best_child is None:
            print("No best child found; stopping.")
            break

        # For reporting, compute the grade of the new prefix (best_child.current_text)
        best_score = grade_prefix(
            llm, tokenizer, grader, device, problem, best_child.current_text
        )

        print(f"Selected chunk (visits={best_child.visit_count}, score={best_score:.4f}):")
        print("--------------------------------------------------")
        print(best_action.strip())
        print()

        current_text = best_child.current_text

        # Stopping conditions
        if best_score > threshold:
            print(
                f"Stopping: Score threshold {threshold} reached (score={best_score:.4f})"
            )
            break

        if best_child.eos_reached:
            print("Stopping: EOS reached in best child.")
            break

        if not best_action.strip():
            print("Stopping: Best action is empty.")
            break

    final_score = grade_prefix(llm, tokenizer, grader, device, problem, current_text)
    return current_text, final_score



# Main

def main():
    parser = argparse.ArgumentParser(description="Guided generation with grading network")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of candidate completions per iteration (root-level for MCTS, all iterations for greedy)",
    )
    parser.add_argument(
        "--max_new",
        type=int,
        default=150,
        help="Max new tokens per iteration",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Score threshold to stop",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=30,
        help="Maximum outer iterations",
    )
    parser.add_argument(
        "--num_simulations",
        type=int,
        default=64,
        help="Number of MCTS simulations per outer iteration (only for mcts)",
    )
    parser.add_argument(
        "--c_puct",
        type=float,
        default=1.0,
        help="PUCT exploration constant (only for mcts)",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help="Maximum depth in MCTS (in chunks, only for mcts)",
    )
    parser.add_argument("--problem", type=str, default=None, help="Math problem to solve")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load LLM
    print("Loading language model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    llm = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, trust_remote_code=True)
    llm.to(device).eval()
    llm_hidden_size = llm.config.hidden_size
    print(f" LLM loaded (hidden_size={llm_hidden_size})")

    # Load grader
    print("Loading grading network")
    grader = load_grading_model(args.checkpoint, device, llm_hidden_size)
    print(" Grading network loaded\n")

    # Define problem
    problem = (
        args.problem
        if args.problem
        #Think you can do this one?
        else "Determine all positive integers n for which there exist positive integers a, b, and c satisfying $2a^n +3b^n = 4c^n$"
    )

    print("=" * 80)
    print(f"Problem: {problem}")
    print("=" * 80)


    final_text, final_score = mcts_guided_generation(
        llm=llm,
        tokenizer=tokenizer,
        grader=grader,
        device=device,
        problem=problem,
        k_root=args.k,
        max_new=args.max_new,
        threshold=args.threshold,
        max_iters=args.max_iters,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        max_depth=args.max_depth,
    )

    print("\n" + "=" * 80)
    print("FINAL OUTPUT")
    print("=" * 80)
    print(final_text)
    print(f"\nFinal grading probability: {final_score:.4f}")


if __name__ == "__main__":
    main()
