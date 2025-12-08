# LLM_Math_Search
# LLM Hidden State Grading for Iterative Mathematical Reasoning

This project trains a feedforward neural network (FFN) to grade mathematical reasoning quality by analyzing the last hidden states from a language model. The grader is then used to guide iterative generation, selecting better reasoning paths at each step.

## Overview

**This repository includes the initial dataset** (`Full_samples.jsonl`) so you can skip the data preparation step. You'll need to download Qwen 2.5 3B locally (instructions below).

The pipeline consists of three main stages:

1. **Hidden State Extraction**: Run inference on a large dataset of mathematical problems and solutions, extracting hidden states from each token position
2. **Last Hidden State Processing**: Extract only the final hidden state from each sequence for efficient training
3. **FFN Training**: Train a binary classifier on these hidden states to predict solution correctness
4. **Guided Generation**: Use the trained grader to iteratively generate and select high-quality reasoning steps

## Key Features

- **Sharded Storage**: Handles 2+ TB datasets efficiently using PyTorch shard files
- **Progressive Prefix Sampling**: Creates training samples from intermediate reasoning steps (every k tokens)
- **Residual FFN Architecture**: Deep feedforward network with residual connections and proper initialization
- **Mixed Precision Training**: FP16 training with gradient accumulation for efficient GPU utilization
- **Iterative Generation**: Beam search variant that grades partial solutions and selects the best continuation

## Project Structure
```
├── Data/
│   └── Full_samples.jsonl      # Provided dataset with math problems and solutions
|   └── Sample_Hard_Problems.csv  #Eval dataset
├── Local_Models/
│   └── qwen3_3b_local/           # Download Qwen 2.5 3B here (see Step 0)
├── Inference_Shard.py          # Extract hidden states from LLM during inference
├── extract_last_hidden.py      # Extract final hidden state from each sequence
├── optimized_dataset.py         # Dataset classes for loading sharded data
├── FFN_Model.py                 # FFN architectures (simple and residual)
├── train_ffn_sharded.py        # Main training script
├── iter_solve.py                # Iterative generation with grading network (Greedy Search)
├── mcts_algo.py                 # (Optional) MCTS-based search algorithm
├── run_guided_generation_batch.py  # Generates Greedy Search Eval dataset
├── math_set_generation_script_GPU.py #Generates Unguided Search Eval dataset
├── model_training_bash_script.txt       # Example SLURM training jobs
└── last_hidden_extraction_bash.txt      # Example SLURM extraction job
```
## Requirements

```bash
pip install torch transformers numpy tqdm
```

**Hardware Requirements:**
- Stage 1 (Inference): A100 GPU (10+ hours for large datasets)
- Stage 2 (Extraction): CPU-only (2-3 hours)
- Stage 3 (Training): A40/A100 GPU (1-6 hours depending on model size)
- Stage 4 (Inference): GPU with 16GB+ VRAM

## Reproduction Guide

### Step 0: Setup

#### Download the Dataset
The initial dataset (`Full_samples.jsonl`) is provided in this repository. Place it in:
```
Data/Full_samples.jsonl
```

**Dataset Schema:**
```json
{"question": "What is 2+2?", "answer": "2+2=4", "is_correct": true}
```

#### Download Qwen 2.5 3B Model

You'll need a local installation of Qwen 2.5 3B. Download it using Hugging Face:

```bash
# Option 1: Using huggingface-cli
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir /path/to/qwen2.5-3b

# Option 2: Using Python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen2.5-3B-Instruct", local_dir="/path/to/qwen2.5-3b")
```

**Update the model path** in the scripts:
- `Inference_Shard.py`: Set `MODEL_ID = "/path/to/qwen2.5-3b"`
- `iter_solve.py`: Set `MODEL_ID = "/path/to/qwen2.5-3b"`

**Note**: The model requires ~7GB of disk space. Make sure you have sufficient storage.

### Step 1: Extract Hidden States

**⚠️ WARNING: This generates 2+ TB of data and takes 10+ hours on an A100**

```bash
python Inference_Shard.py
```

**Configuration** (edit in script):
- `MODEL_ID`: Path to your local LLM (e.g., Qwen 2.5 3B)
- `INPUT_PATH`: Path to your JSONL dataset
- `OUT_DIR`: Output directory for shards
- `BATCH_SIZE`: Adjust based on GPU memory
- `SHARD_SIZE`: Samples per shard file (default: 2000)

**Output**: Creates `qwen3_hidden_states_open_math_sharded/` with:
- `shard_XXXX.pt`: Hidden state shards
- `catalog.jsonl`: Metadata for each sample

### Step 2: Extract Last Hidden States

This dramatically reduces storage requirements by keeping only the final token's hidden state:

```bash
python extract_last_hidden.py \
    --input_dir qwen3_hidden_states_open_math_sharded \
    --output_dir qwen3_last_hidden_sharded \
    --samples_per_shard 25000 \
    --stride_k 100 \
    --max_length 1024
```

**Arguments:**
- `--stride_k`: Create samples every k tokens (progressive prefixes)
- `--samples_per_shard`: Samples per output shard (higher = fewer files)
- `--max_length`: Maximum sequence length to consider

**Output**: Creates `qwen3_last_hidden_sharded/` with batched tensor shards

### Step 3: Train the Grading Model

```bash
python train_ffn_sharded.py \
    --train_data qwen3_last_hidden_sharded \
    --output_dir checkpoints_ffn \
    --use_residual \
    --use_layer_norm \
    --hidden_dim 2048 \
    --num_layers 10 \
    --batch_size 4096 \
    --epochs 10 \
    --use_amp \
    --accumulation_steps 4 \
    --val_split 0.1
```

**Key Arguments:**
- `--use_residual`: Use residual FFN architecture (recommended)
- `--hidden_dim`: Hidden dimension for residual blocks
- `--num_layers`: Number of residual blocks
- `--use_amp`: Enable mixed precision training
- `--accumulation_steps`: Gradient accumulation steps
- `--val_split`: Fraction of data for validation

**Training Tips:**
- Start with 10 layers and 2048 hidden dim
- Batch size of 4096 works well on A40/A100
- Enable `--use_layer_norm` for stability
- Model selection is based on validation F1 score

**Output**: 
- `checkpoints_ffn/best_model.pt`: Best model checkpoint
- `checkpoints_ffn/history.json`: Training metrics

### Step 4: Generate Evaluation Data



#### 4a. Run Unguided Baseline
```bash
python math_set_generation_script_GPU.py
```

**What it does:**
- Loads problems from `Data/Sample_Hard_Problems.csv`
- Generates solutions using the base LLM
- Saves sharded outputs to `qwen3_hidden_states_sample_problems/` in same schema as intial inference

**Configuration** (edit in script):
- `MATH_PROBLEMS`: Adjust the slice `[0:1000]` to control number of problems
- `MAX_NEW_TOKENS`: Maximum tokens to generate per solution (default: 2048)
- `TEMPERATURE`: Sampling temperature (default: 0.7)
- `BATCH_SIZE`: Adjust based on GPU memory

**Output**: Creates sharded hidden states for evaluation

#### 4b. Run Batch Guided Generation
```bash
python run_guided_generation_batch.py \
    --checkpoint checkpoints_ffn/best_model.pt \
    --num_problems 1000 \
    --k 5 \
    --max_new 150 \
    --threshold 0.95 \
    --max_iters 30 \
    --output_dir guided_generation_results \
    --verbose
```

**Arguments:**
- `--num_problems`: Number of problems to process from the CSV
- `--k`: Number of candidate completions per iteration
- `--max_new`: Maximum new tokens per iteration
- `--threshold`: Score threshold to stop generation (0-1)
- `--max_iters`: Maximum number of iterations
- `--verbose`: Print detailed logs for each problem

**How it works:**
1. Loads problems from `Data/Sample_Hard_Problems.csv`
2. For each problem:
   - Generate k candidate continuations
   - Grade each candidate using the FFN
   - Select the best one and continue
   - Stop when score exceeds threshold or EOS is generated
3. Saves results incrementally to JSONL file

**Output**: 
- `guided_generation_results/results_TIMESTAMP.jsonl`: Individual problem results
- `guided_generation_results/summary_TIMESTAMP.json`: Aggregate statistics

### Step 5:

You now have the completed final evaluation dataset


## Model Architectures

### Simple FFN (`FFNBinaryClassifier`)
- Stacked linear layers with ReLU activation
- Optional batch normalization
- Dropout for regularization

### Residual FFN (`ResidualFFNBinaryClassifier`) - **Recommended**
- Residual blocks with layer normalization
- Learnable residual scaling
- 4x expansion in hidden dimension
- Xavier initialization with small output weights
- GELU activation

## Dataset Classes

- `ShardedHiddenStateDataset`: Full sequence hidden states (slow, not recommended for training)
- `LazyShardedHiddenStateDataset`: Lazy cache variant of above, **USE THIS**
- `LastHiddenFullDataset`: **Recommended** - Loads all last hidden states into RAM

## Training Configuration Examples

See `model_training_bash_script.txt` for various configurations:
- 5-25 layer models
- 1024-3072 hidden dimensions
- Different dropout rates
- All trained for 10 epochs with batch size 4096

## Batch Script Examples

For SLURM clusters, see:
- `last_hidden_extraction_bash.txt`: CPU job for extraction
- `model_training_bash_script.txt`: GPU job for training multiple models sequentially, exactly the same setup as project

## Performance Notes

**Storage:**
- Full hidden states: ~2 TB for 1M sequences
- Last hidden only: ~30 GB for 1M sequences

**Training Speed:**
- ~1-2 hours for 10-layer model on A100
- ~4-6 hours for 20-layer model on A40

**Inference:**
- Guided generation: ~2-5 seconds per iteration (k=5)
- Depends heavily on model size and max_new_tokens




