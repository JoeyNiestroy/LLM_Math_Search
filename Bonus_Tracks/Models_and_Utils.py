

"""
This code contains the model class for a Transformer Encoder built off the llama arch. Code is consistant with the llama hugging face repo except for the attention mechansim.

Forward pass expects raw hidden states not input_ids like an LLM, this is built to be use case specific.  
"""




from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Dict, Any, Iterable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



# Utils for model, somewhat consisent with modern libraries/implementations. Some copy paste some AI here. 
# 
class RMSNorm(nn.Module):
    """Root Mean Square LayerNorm (no mean subtraction), like LLaMA."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Helper for RoPE: split last dim in half and rotate."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """
    RoPE as used in LLaMA: apply to q and k.
    Stores cos/sin caches up to max_seq_len during first call or when extended.
    """
    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 2048, device=None, dtype=None):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.base = base
        self.register_buffer("cos_cache", None, persistent=False)
        self.register_buffer("sin_cache", None, persistent=False)
        self.max_seq_len_cached = 0
        self.default_device = device
        self.default_dtype = dtype

    def _build_cache(self, seq_len: int, device, dtype):
        if self.max_seq_len_cached >= seq_len:
            return
        dim = self.head_dim
        # shape: [dim/2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
        # positions: [seq_len, 1]
        t = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)  # [T, 1]
        # freqs: [T, dim/2]
        freqs = t * inv_freq.unsqueeze(0)  # [T, dim/2]
        # build cos/sin with interleaving
        emb = torch.cat([freqs, freqs], dim=-1)  # [T, dim]
        self.cos_cache = emb.cos()  # [T, dim]
        self.sin_cache = emb.sin()  # [T, dim]
        self.max_seq_len_cached = seq_len

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, n_heads, T, head_dim] (only used for shape/device/dtype)
        returns cos, sin of shape [1, 1, T, head_dim] to broadcast on q/k
        """
        b, h, t, d = x.shape
        device = x.device if self.default_device is None else self.default_device
        dtype = x.dtype if self.default_dtype is None else self.default_dtype
        self._build_cache(start_pos + t, device, dtype)
        cos = self.cos_cache[start_pos:start_pos + t].unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
        sin = self.sin_cache[start_pos:start_pos + t].unsqueeze(0).unsqueeze(0)  # [1,1,T,D]
        return cos.to(x.dtype), sin.to(x.dtype)

    @staticmethod
    def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q,k: [B, H, T, D], cos/sin: [1, 1, T, D]
        q_rot = (q * cos) + (_rotate_half(q) * sin)
        k_rot = (k * cos) + (_rotate_half(k) * sin)
        return q_rot, k_rot


class SwiGLU(nn.Module):
    """
    LLaMA-style MLP: two linears (gate/up) -> SiLU(gate) * up -> down
    No biases by default.
    """
    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up   = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# Attention, simple SDPA, qkv is one matrix vs 3 , non casual uses additive mask 

class LlamaStyleAttention(nn.Module):
    """
    Multi-Head Attention w/ RoPE.
    - No biases (like LLaMA)
    - Uses PyTorch SDPA when available
    - Bidirectional (encoder), i.e., no causal mask applied here
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope: RotaryEmbedding,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rope = rope

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                 # [B, T, C]
        attention_mask: Optional[torch.Tensor] = None,  # [B, T] bool (True = keep, False = pad) or additive float mask broadcastable to [B, 1, T, T]
        start_pos: int = 0               # for chunked encoding with RoPE
    ) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = qkv.split(c, dim=-1)  # each [B, T, C]

        # reshape to heads
        def shape_heads(z):
            return z.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

        q = shape_heads(q)
        k = shape_heads(k)
        v = shape_heads(v)

        # RoPE on q,k
        cos, sin = self.rope(q, start_pos=start_pos)  # [1,1,T,D]
        q, k = RotaryEmbedding.apply_rope(q, k, cos, sin)

        # scaled dot-product attention
        # SDPA expects [B,H,T,D]
        if attention_mask is not None:
            # Normalize mask:
            # If bool mask provided: True = keep token; False = pad -> we convert to additive mask
            if attention_mask.dtype == torch.bool:
                # Build [B, 1, 1, T_k] then broadcast to [B, H, T_q, T_k]
                keep = attention_mask[:, None, None, :]  # True where valid
                # additive mask: 0 for valid, -inf for invalid
                add_mask = (~keep) * torch.finfo(q.dtype).min
            else:
                # assume float additive mask broadcastable to [B, 1, T, T]
                add_mask = attention_mask

            # SDPA supports additive mask (same dtype as q/k/v), broadcastable to [B, H, T, T]
            # We handle broadcasting above, I think pytorch does same conversion if you pass them boolean mask
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=add_mask,  # additive
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        else:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        # shape back
        attn = attn.transpose(1, 2).contiguous().view(b, t, c)  # [B, T, C]
        return self.out_proj(attn)



# Encoder Block, follows LLama arch, double pre norm and res connections

class LlamaStyleEncoderBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rope: RotaryEmbedding,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn  = LlamaStyleAttention(hidden_size, num_heads, rope, dropout=attn_dropout, bias=False)
        self.drop1 = nn.Dropout(resid_dropout)

        self.norm2 = RMSNorm(hidden_size)
        self.mlp   = SwiGLU(hidden_size, intermediate_size, bias=False)
        self.drop2 = nn.Dropout(resid_dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, start_pos: int = 0) -> torch.Tensor:
        # Pre-norm -> Attn -> Residual
        h = self.norm1(x)
        h = self.attn(h, attention_mask=attention_mask, start_pos=start_pos)
        x = x + self.drop1(h)

        # Pre-norm -> MLP -> Residual
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + self.drop2(h)
        return x



# Config & Encoder , nothing crazy here TODO check outstanding issues and implement functionaility 

@dataclass
class LlamaStyleEncoderConfig:
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    intermediate_size: int = 1536  # ~ 3x hidden; LLaMA uses ~ (4/3)*4x via SwiGLU; tune as you like
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    vocab_size: int = 0  # encoder doesn't need this unless you add embeddings here
    pad_token_id: Optional[int] = None  # if you want padding mask helper


class LlamaStyleEncoder(nn.Module):
    """
    Bidirectional Transformer Encoder, LLaMA-ish:
    - Stack of pre-norm blocks with RoPE attention + SwiGLU MLP
    - No token embedding layer included (pass in your continuous inputs)
    """
    def __init__(self, cfg: LlamaStyleEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.rope = RotaryEmbedding(
            head_dim=cfg.hidden_size // cfg.num_heads,
            base=cfg.rope_base,
            max_seq_len=cfg.max_seq_len,
        )
        self.layers = nn.ModuleList([
            LlamaStyleEncoderBlock(
                hidden_size=cfg.hidden_size,
                num_heads=cfg.num_heads,
                intermediate_size=cfg.intermediate_size,
                rope=self.rope,
                attn_dropout=cfg.attn_dropout,
                resid_dropout=cfg.resid_dropout,
            )
            for _ in range(cfg.num_layers)
        ])
        self.final_norm = RMSNorm(cfg.hidden_size)

    @staticmethod
    def make_key_padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
        """
        Build boolean mask [B, T] where True=keep, False=pad.
        input_ids: [B, T] (integers). If you pass continuous x instead of ids, build your own mask.
        """
        return ~(input_ids == pad_token_id)

    def forward(
        self,
        x: torch.Tensor,                              # [B, T, C] continuous inputs (e.g., from your own embedding)
        attention_mask: Optional[torch.Tensor] = None # [B, T] bool (True=keep), or additive float mask broadcastable to [B,1,T,T]
    ) -> torch.Tensor:
        # Clamp sequence length for RoPE cache (auto-extends if needed)
        _, T, _ = x.shape
        if T > self.cfg.max_seq_len:
            # Will still work: rope cache auto-extends, but you may want to increase cfg.max_seq_len.
            pass

        h = x
        for layer in self.layers:
            h = layer(h, attention_mask=attention_mask, start_pos=0)
        h = self.final_norm(h)
        return h
    





# Binary Classification head wrapper for encoder, probably could be just wrapped in orignal class, but allows clear CLS vs averging logic. Adapter not expected to be used

class AdapterIn(nn.Module):
    """Optional pre-norm + projection if upstream dim != encoder hidden_size."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.norm = RMSNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
    def forward(self, x):  # [B,T,in_dim]
        return self.proj(self.norm(x))

def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """x:[B,T,C], mask:[B,T] bool -> [B,C]"""
    m = mask.float().unsqueeze(-1)                # [B,T,1]
    denom = m.sum(dim=1).clamp_min(1e-6)         # [B,1]
    return (x * m).sum(dim=1) / denom

class LlamaStyleBinaryClassifier(nn.Module):
    def __init__(
        self,
        encoder: LlamaStyleEncoder,
        in_dim: int,                  # upstream hidden size
        use_cls_token: bool = False,  # if True, prepends trainable [CLS]
        dropout: float = 0.0
    ):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = encoder.cfg.hidden_size

        # optional adapter if upstream dim != encoder hidden size
        self.adapter_in = None
        if in_dim != self.hidden_size:
            self.adapter_in = AdapterIn(in_dim, self.hidden_size)

        self.use_cls = use_cls_token
        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            nn.init.normal_(self.cls, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, 1, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,                 # [B,T,in_dim] from upstream
        attention_mask: torch.Tensor,                # [B,T] bool, True=keep, False=pad
        labels: torch.Tensor = None                  # [B,1] float (0/1) or {0,1} long
    ):
        x = hidden_states
        if self.adapter_in is not None:
            x = self.adapter_in(x)                   # [B,T,H]

        # add [CLS] if requested
        if self.use_cls:
            B = x.size(0)
            cls_tok = self.cls.expand(B, -1, -1)     # [B,1,H]
            x = torch.cat([cls_tok, x], dim=1)       # [B,T+1,H]
            # prepend True to mask for CLS so it participates in attention
            cls_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([cls_mask.bool(), attention_mask.bool()], dim=1)

        # encode (bidirectional)
        enc = self.encoder(x, attention_mask=attention_mask)  # [B,T(±1),H]

        # pool
        if self.use_cls:
            pooled = enc[:, 0, :]                     # [CLS]
        else:
            pooled = masked_mean_pool(enc, attention_mask)     # masked mean

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)  # [B]
        probs = torch.sigmoid(logits)

        out = {"logits": logits, "probs": probs}

        if labels is not None:
            # We need to flatten [B,1] down to [B]
            labels = labels.view(-1)  # Flatten to [B] regardless of input shape
            if labels.dtype != torch.float32 and labels.dtype != torch.float16 and labels.dtype != torch.bfloat16:
                labels = labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            out["loss"] = loss

        return out




#Old code, deprc after the move to shards 

# class HiddenStateDataset(Dataset):    
#         """
#     Loads (hidden_states, label) pairs and generates multiple samples per sequence
#     by taking prefixes at regular intervals (stride_k).
    
#     Each original sequence of length L generates floor(L/stride_k) samples:
#       - H[0:k, :]
#       - H[0:2k, :]
#       - ...
#       - H[0:floor(L/stride_k)*k, :]
    
#     Optionally includes the full sequence even if L is not a multiple of stride_k.
#     """
#         def __init__(
#             self,
#             root: str | Path,
#             stride_k: int = 100,
#             max_length: Optional[int] = None,  # Add this parameter
#             include_final: bool = True,
#             include_input_ids: bool = True,
#             extra_meta_fields: Optional[Sequence[str]] = None,
#             lazy_cache: bool = False,
#         ):
#             self.root = Path(root)
#             self.stride_k = stride_k
#             self.max_length = max_length  # Store it
#             self.include_final = include_final
#             self.include_input_ids = include_input_ids
#             self.extra_meta_fields = set(extra_meta_fields or [])
#             self.lazy_cache = lazy_cache
            
#             # ... rest of __init__ stays the same ...
            
#             cat_path = self.root / "catalog.jsonl"
#             assert cat_path.exists(), f"Missing {cat_path}"
            
#             self._meta: List[Dict[str, Any]] = []
#             with open(cat_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     if line.strip():
#                         m = json.loads(line)
#                         for k in ("id", "length", "label_correct"):
#                             assert k in m, f"catalog row missing '{k}'"
#                         self._meta.append(m)
            
#             # Build index with max_length constraint
#             self._samples: List[Tuple[int, int]] = []
#             for meta_idx, m in enumerate(self._meta):
#                 L = int(m["length"])
                
#                 # Cap L at max_length if specified
#                 effective_L = min(L, self.max_length) if self.max_length else L
                
#                 # Generate samples at k, 2k, 3k, ... up to effective_L
#                 for multiplier in range(1, (effective_L // self.stride_k) + 1):
#                     prefix_len = multiplier * self.stride_k
#                     self._samples.append((meta_idx, prefix_len))
                
#                 # Optionally add the full sequence (up to max_length)
#                 if self.include_final:
#                     last_included = (effective_L // self.stride_k) * self.stride_k
#                     if last_included < effective_L:
#                         self._samples.append((meta_idx, effective_L))
            
#             self._cache_hidden: Dict[str, torch.Tensor] = {}
#             self._cache_ids: Dict[str, torch.Tensor] = {}

#         def __len__(self) -> int:
#             return len(self._samples)

#         def _load_hidden(self, item_id: str) -> torch.Tensor:
#             if self.lazy_cache and item_id in self._cache_hidden:
#                 return self._cache_hidden[item_id]
#             t = torch.load(self.root / f"{item_id}_hidden.pt", map_location="cpu")
#             if self.lazy_cache:
#                 self._cache_hidden[item_id] = t
#             return t

#         def _load_ids(self, item_id: str) -> Optional[torch.Tensor]:
#             if not self.include_input_ids:
#                 return None
#             if self.lazy_cache and item_id in self._cache_ids:
#                 return self._cache_ids[item_id]
#             p = self.root / f"{item_id}_input_ids.pt"
#             if p.exists():
#                 t = torch.load(p, map_location="cpu")
#                 if self.lazy_cache:
#                     self._cache_ids[item_id] = t
#                 return t
#             return None

#         def __getitem__(self, idx: int) -> Dict[str, Any]:
#             meta_idx, prefix_len = self._samples[idx]
#             m = self._meta[meta_idx]
#             item_id: str = m["id"]
            
#             # Load full hidden states and slice to prefix
#             hidden_full = self._load_hidden(item_id)  # [L_full, H]
#             hidden = hidden_full[:prefix_len, :]      # [prefix_len, H]
            
#             input_ids = None
#             if self.include_input_ids:
#                 input_ids_full = self._load_ids(item_id)
#                 if input_ids_full is not None:
#                     input_ids = input_ids_full[:prefix_len]
            
#             y = torch.tensor([float(m["label_correct"])], dtype=torch.float32)
            
#             out: Dict[str, Any] = {
#                 "hidden_states": hidden,
#                 "attention_mask": torch.ones(prefix_len, dtype=torch.bool),
#                 "length": prefix_len,
#                 "label": y,
#                 "id": f"{item_id}_prefix{prefix_len}",
#                 "original_id": item_id,
#                 "original_length": int(m["length"]),
#             }
            
#             if input_ids is not None:
#                 out["input_ids"] = input_ids
            
#             for k in self.extra_meta_fields:
#                 if k in m:
#                     out[k] = m[k]
            
#             return out
        
# def collate_hidden_states(batch):
#     """
#     Collates variable-length hidden state sequences by padding.
#     """
#     # Extract fields
#     hidden_states = [item['hidden_states'] for item in batch]  # List of [L_i, H]
#     attention_masks = [item['attention_mask'] for item in batch]  # List of [L_i]
#     labels = torch.stack([item['label'] for item in batch])  # [B, 1]
#     lengths = torch.tensor([item['length'] for item in batch])  # [B]
#     ids = [item['id'] for item in batch]
#     original_ids = [item['original_id'] for item in batch]
#     original_lengths = torch.tensor([item['original_length'] for item in batch])
    
#     # Pad hidden states: [B, max_L, H]
#     hidden_states_padded = pad_sequence(hidden_states, batch_first=True, padding_value=0.0)
    
#     # Pad attention masks: [B, max_L]
#     attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    
#     result = {
#         'hidden_states': hidden_states_padded,
#         'attention_mask': attention_masks_padded,
#         'length': lengths,
#         'label': labels,
#         'id': ids,
#         'original_id': original_ids,
#         'original_length': original_lengths,
#     }
    
#     # Handle input_ids if present
#     if 'input_ids' in batch[0] and batch[0]['input_ids'] is not None:
#         input_ids = [item['input_ids'] for item in batch]
#         input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
#         result['input_ids'] = input_ids_padded
    
#     # Handle extra metadata fields
#     for key in batch[0].keys():
#         if key not in result and key not in ['hidden_states', 'attention_mask', 'length', 
#                                                'label', 'id', 'original_id', 'original_length', 'input_ids']:
#             # Just collect as list for metadata
#             result[key] = [item[key] for item in batch]
    
#     return result
