"""
Extract only the last hidden state from each sequence and save to new shards.
"""
import json
from pathlib import Path
import torch
from tqdm import tqdm
import argparse

# Import your existing dataset
from optimized_dataset import LazyShardedHiddenStateDataset


# Shard Writer for Last Hidden States
# functions pretty similar to the other shard writer
#Everything is the same size so they saved to large matrices
class LastHiddenShardWriter:
    """
    Shard writer that batches last hidden states into tensors.
    """
    def __init__(self, out_dir: Path, samples_per_shard: int, dtype=torch.float16):
        self.out_dir = out_dir
        self.samples_per_shard = samples_per_shard
        self.dtype = dtype

        # Buffers
        self.hidden_buffer = []
        self.label_buffer = []
        self.meta_buffer = []

        self.shard_idx = 0
        self.global_sample_idx = 0
        self.catalog = []


    def add_sample(self, last_hidden, label, metadata):
        """Accumulate one sample until shard is full."""
        last_hidden = last_hidden.to(self.dtype).cpu()
        label = label.float().cpu()

        self.hidden_buffer.append(last_hidden)
        self.label_buffer.append(label)
        self.meta_buffer.append(metadata)

        # Add to catalog
        self.catalog.append({
            'sample_id': f"sample_{self.global_sample_idx:07d}",
            'shard_file': f"shard_{self.shard_idx:04d}.pt",
            'shard_idx': self.shard_idx,
            'local_idx': len(self.hidden_buffer) - 1,
            'label': float(label.item()),
            'original_id': metadata['original_id'],
            'prefix_length': metadata['prefix_length'],
        })
        self.global_sample_idx += 1

        if len(self.hidden_buffer) >= self.samples_per_shard:
            self._write_shard()


    def _write_shard(self):
        """Write one shard as batched tensors."""
        if not self.hidden_buffer:
            return

        shard_path = self.out_dir / f"shard_{self.shard_idx:04d}.pt"

        # Stack tensors
        last_hiddens = torch.stack(self.hidden_buffer)            # [N, H]
        labels = torch.stack(self.label_buffer).view(-1)          # [N]
        metadatas = list(self.meta_buffer)                        # list of dicts

        torch.save(
            {
                "last_hiddens": last_hiddens,
                "labels": labels,
                "metadata": metadatas,
            },
            shard_path,
        )

        print(f"Wrote shard {self.shard_idx} with {len(self.hidden_buffer)} samples "
              f"({last_hiddens.shape[0]}×{last_hiddens.shape[1]})")

        # Reset buffers
        self.hidden_buffer.clear()
        self.label_buffer.clear()
        self.meta_buffer.clear()
        self.shard_idx += 1


    def finalize(self):
        """Flush remaining samples and write catalog/summary."""
        if self.hidden_buffer:
            self._write_shard()

        # Catalog
        catalog_path = self.out_dir / "catalog.jsonl"
        with open(catalog_path, "w", encoding="utf-8") as f:
            for entry in self.catalog:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        summary_path = self.out_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "total_samples": len(self.catalog),
                    "total_shards": self.shard_idx,
                    "samples_per_shard": self.samples_per_shard,
                    "dtype": str(self.dtype),
                },
                f,
                indent=2,
            )

        print(f"\n Wrote catalog with {len(self.catalog):,} samples across "
              f"{self.shard_idx} shards.")
        return len(self.catalog), self.shard_idx



# Main processing
def extract_last_hidden_states(args):
    print("="*80)
    print("EXTRACTING LAST HIDDEN STATES TO NEW SHARDS")
    print("="*80)
    
    INPUT_DIR = Path(args.input_dir)
    OUTPUT_DIR = Path(args.output_dir)
    
    print(f"\n Input:  {INPUT_DIR}")
    print(f" Output: {OUTPUT_DIR}")
    print(f" Samples per shard: {args.samples_per_shard}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # Load dataset
    print(f"\n Loading dataset...")
    dataset = LazyShardedHiddenStateDataset(
        root=INPUT_DIR,
        stride_k=args.stride_k,
        max_length=args.max_length,
        max_cached_shards=args.max_cached_shards,
    )
    print(f" Dataset loaded: {len(dataset):,} samples")
    
    # Initialize shard writer
    shard_writer = LastHiddenShardWriter(OUTPUT_DIR, args.samples_per_shard)
    
    # Process samples
    print(f"\n Extracting last hidden states...")
    
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        sample = dataset[idx]
        
        # Extract last hidden state
        hidden_states = sample['hidden_states']  # [L, H]
        attention_mask = sample['attention_mask']  # [L]
        label = sample['label']  # scalar or [1]
        
        # Find last non-padded position
        # attention_mask is True for valid tokens
        # should straight 1s since we iter 1by1 but just in case
        length = attention_mask.sum().item()
        last_idx = length - 1
        
        # Get last hidden state
        last_hidden = hidden_states[last_idx]  # [H]
        
        # Prepare metadata
        metadata = {
            'original_id': sample['original_id'],
            'sample_id': sample['id'],
            'prefix_length': sample['length'],
            'original_length': sample['original_length'],
        }
        
        # Add to shard
        shard_writer.add_sample(last_hidden, label, metadata)
        
        # Progress bar
        if (idx + 1) % 100_000 == 0:
            print(f"   Processed {idx + 1:,} samples...")
        
    
    # Finalize
    total_saved, num_shards = shard_writer.finalize()
    
    print(f"\n Saved {total_saved:,} samples to {num_shards} shard files")
    
    # Calculate storage stats
    print(f"\n{'='*80}")
    print("STORAGE STATISTICS")
    print(f"{'='*80}")
    
    def get_dir_size(path):
        total = 0
        for f in path.glob("*.pt"):
            total += f.stat().st_size
        return total

    output_size_bytes = get_dir_size(OUTPUT_DIR)
    output_size_gb = output_size_bytes / (1024**3)
    
    print(f"Output size: {output_size_gb:.2f} GB")    
    avg_shard_size_mb = (output_size_bytes / num_shards) / (1024**2)
    print(f"\nAvg shard size: {avg_shard_size_mb:.2f} MB")
    print(f"Total samples: {total_saved:,}")
    print(f"Total shards: {num_shards}")
    
    # Test loading speeds 
    print(f"\n{'='*80}")
    print("TESTING LOAD SPEED")
    print(f"{'='*80}")
    
    import time
    
    # Test loading a shard
    test_shard = OUTPUT_DIR / "shard_0000.pt"
    
    print(f"\nLoading {test_shard}...")
    t0 = time.time()
    data = torch.load(test_shard, map_location='cpu', weights_only=False)
    t_load = time.time() - t0

    N = data["last_hiddens"].shape[0]
    print(f" Loaded shard with {N} samples in {t_load:.3f}s")
    print(f"   Time per sample: {t_load / N:.6f}s")

    # Inspect first sample
    i = 0
    print(f"\n Sample structure:")
    print(f"   last_hidden shape: {data['last_hiddens'][i].shape}")
    print(f"   label: {data['labels'][i]}")
    print(f"   metadata: {data['metadata'][i]}")
    

    print(f"\nDataset saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, 
                       default="qwen3_hidden_states_open_math_sharded",
                       help="Input directory with full sequence shards")
    parser.add_argument("--output_dir", type=str, 
                       default="qwen3_last_hidden_sharded",
                       help="Output directory for last hidden state shards")
    parser.add_argument("--samples_per_shard", type=int, default=10000,
                       help="Number of samples per shard file")
    parser.add_argument("--stride_k", type=int, default=100,
                       help="Stride for progressive prefixes")
    parser.add_argument("--max_length", type=int, default=1024,
                       help="Maximum sequence length")
    #This doesn't really matter bc we don't care about randomly sampling
    parser.add_argument("--max_cached_shards", type=int, default=1,
                       help="Max shards to cache during processing")
    
    args = parser.parse_args()
    extract_last_hidden_states(args)
