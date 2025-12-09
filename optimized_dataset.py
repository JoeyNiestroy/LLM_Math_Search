"""
'Optimized' dataset loader for sharded hidden states.

LazyShardedHiddenStateDataset and ShardedHiddenStateDataset are unused classes for inference using FULL SEQUENCE hidden states. SEVERE speed issues, don't use them for training models unless this text isn't here
They were intially designed for a sampler that takes adavantage of the lazy cache behavior, even with that they are awful. They are used in the building the last hidden dataset though bc that is off GPU. 
They also require a collate function to work properly, see the extract_last_hidden for more details. It returns the 

LastHiddenFullDataset is a beautiful class that works perfectly 
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset
from tqdm import tqdm



#This is the dataset class that is designed to work with the schema from inference_shard
#It includes the logic for prefix sampling, it is not computed until indexed for that sample 
#Part of the speed issue comes from the pickling/unpickling the dict objects
#The other half is the padding logic needed to deal with full the full sequences
class ShardedHiddenStateDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        stride_k: int = 100,
        max_length: Optional[int] = None,
        include_final: bool = True,
        preload_shards: bool = True,
    ):
        """
        Args:
            root: Directory containing sharded data and catalog.jsonl
            stride_k: Create samples every k tokens (progressive prefixes)
            max_length: Maximum sequence length to consider
            include_final: Include full sequence even if not multiple of stride_k
            preload_shards: Load all shards into memory at init (faster, uses more RAM)
        """
        self.root = Path(root)
        self.stride_k = stride_k
        self.max_length = max_length
        self.include_final = include_final
        self.preload_shards = preload_shards
        
        # Load catalog
        cat_path = self.root / "catalog.jsonl"
        assert cat_path.exists(), f"Missing {cat_path}"
        
        self._catalog: List[Dict[str, Any]] = []
        with open(cat_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self._catalog.append(json.loads(line))
        
        # Build progressive sample index
        self._samples: List[tuple] = []  # (catalog_idx, prefix_length)
        
        for cat_idx, entry in enumerate(self._catalog):
            L = int(entry["length"])
            
            # Cap at max_length if specified
            effective_L = min(L, self.max_length) if self.max_length else L
            
            # Generate samples at k, 2k, 3k, ...
            for multiplier in range(1, (effective_L // self.stride_k) + 1):
                prefix_len = multiplier * self.stride_k
                self._samples.append((cat_idx, prefix_len))
            
            # Add final sequence if requested
            if self.include_final:
                last_included = (effective_L // self.stride_k) * self.stride_k
                if last_included < effective_L:
                    self._samples.append((cat_idx, effective_L))
        
        # Shard cache
        self._shard_cache: Dict[str, List[Dict]] = {}
        
        # Preload all shards if requested
        if self.preload_shards:
            self._preload_all_shards()
    
    def _preload_all_shards(self):
        """Load all shard files into memory."""
        unique_shards = set(entry['shard_file'] for entry in self._catalog)
        
        print(f"Preloading {len(unique_shards)} shards into memory...")
        for shard_file in unique_shards:
            self._load_shard(shard_file)
        print("All shards loaded!")
    
    def _load_shard(self, shard_file: str) -> List[Dict]:
        """Load a shard file if not already cached."""
        if shard_file in self._shard_cache:
            return self._shard_cache[shard_file]
        
        shard_path = self.root / shard_file
        shard_data = torch.load(shard_path, map_location='cpu')
        self._shard_cache[shard_file] = shard_data
        return shard_data
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cat_idx, prefix_len = self._samples[idx]
        entry = self._catalog[cat_idx]
        
        # Load shard and get sample
        shard_file = entry['shard_file']
        local_idx = entry['local_idx']
        
        shard_data = self._load_shard(shard_file)
        sample = shard_data[local_idx]
        
        # Extract hidden states and input_ids (full sequences)
        hidden_full = sample['hidden_states']  # [L_full, H]
        input_ids_full = sample['input_ids']   # [L_full]
        
        # Slice to prefix length
        hidden = hidden_full[:prefix_len]      # [prefix_len, H]
        input_ids = input_ids_full[:prefix_len]  # [prefix_len]
        
        # Build output
        label = torch.tensor([float(entry["label_correct"])], dtype=torch.float32)
        
        return {
            "hidden_states": hidden,
            "attention_mask": torch.ones(prefix_len, dtype=torch.bool),
            "input_ids": input_ids,
            "length": prefix_len,
            "label": label,
            "id": f"{entry['id']}_prefix{prefix_len}",
            "original_id": entry['id'],
            "original_length": int(entry['length']),
        }


class LazyShardedHiddenStateDataset(ShardedHiddenStateDataset):
    """
    Variant that keeps only recently-used shards in memory
    """
    def __init__(
        self,
        root: str | Path,
        stride_k: int = 100,
        max_length: Optional[int] = None,
        include_final: bool = True,
        max_cached_shards: int = 10,
    ):
        # Don't preload
        super().__init__(
            root=root,
            stride_k=stride_k,
            max_length=max_length,
            include_final=include_final,
            preload_shards=False,
        )
        self.max_cached_shards = max_cached_shards
        self._access_order: List[str] = []
    
    def _load_shard(self, shard_file: str) -> List[Dict]:
        """Load shard with LRU eviction."""
        if shard_file in self._shard_cache:
            # Move to end (most recently used)
            self._access_order.remove(shard_file)
            self._access_order.append(shard_file)
            return self._shard_cache[shard_file]
        
        # Load new shard
        shard_path = self.root / shard_file
        shard_data = torch.load(shard_path, map_location='cpu')
        
        # Add to cache
        self._shard_cache[shard_file] = shard_data
        self._access_order.append(shard_file)
        
        # Evict oldest if cache is full
        if len(self._shard_cache) > self.max_cached_shards:
            oldest = self._access_order.pop(0)
            del self._shard_cache[oldest]
        
        return shard_data



class LastHiddenFullDataset(Dataset):
    """
    Loads all last-hidden-state shards into RAM once and provides
    indexed access to (hidden, label, metadata) triples.
    """
    def __init__(self, root: str | Path, dtype=torch.float16, verbose=True):
        self.root = Path(root)
        self.dtype = dtype


        # Load all shards
        shard_paths = sorted(self.root.glob("shard_*.pt"))
        if verbose:
            print(f"Loading {len(shard_paths)} shards from {self.root} ...")


        #There is no limiting caching here since this whole dataset is designed to avoid IO problems from larger datasets
        all_hiddens, all_labels, all_metas = [], [], []
        for shard_path in tqdm(shard_paths, disable=not verbose):
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
            all_hiddens.append(data["last_hiddens"].to(dtype))
            all_labels.append(data["labels"].float())
            all_metas.extend(data["metadata"])

        # Concatenate along batch dimension
        self.last_hiddens = torch.cat(all_hiddens, dim=0)   # [N, H]
        self.labels = torch.cat(all_labels, dim=0)           # [N]
        self.metadata = all_metas
        self.n, self.h = self.last_hiddens.shape

        if verbose:
            gb = self.last_hiddens.nbytes / (1024**3)
            print(f" Loaded total {self.n:,} samples ({self.h}-dim), "
                  f"{gb:.2f} GB in memory")

        # If this hits you're fucked, something is wrong with extract last hidden
        assert len(self.labels) == len(self.metadata) == self.n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "hidden": self.last_hiddens[idx],
            "label": self.labels[idx],
            "metadata": self.metadata[idx],
        }

    def to_device(self, device):
        #This is only here bc my memory usage was low on the larger GPU, will probably throw OOM on anything but 40gb +
        self.last_hiddens = self.last_hiddens.to(device, non_blocking=True)
        self.labels = self.labels.to(device, non_blocking=True)
        return self
