import torch 
import torch.nn as nn
from .scaled_dot_product import ScaledDotProductAttention
from .causal import causal_mask
from .rope import apply_rotary_emb



class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module for Decoder-only (causal) Transformer models.
    """ 
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads 

        # linear projections for Query, Key, Value, and Output
        self.w_q = nn.Linear(d_model, d_model, bias = False)
        self.w_k = nn.Linear(d_model, d_model, bias = False)
        self.w_v = nn.Linear(d_model, d_model, bias = False)
        self.w_o = nn.Linear(d_model, d_model, bias = False)

        self.attention = ScaledDotProductAttention(dropout_p = dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
        is_causal: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            freqs_cis: Precomputed RoPE frequencies sliced to current sequence positions
            kv_cache: Optional tuple of (cached_k, cached_v) for inference
            is_causal: Whether to apply causal mask
            
        Returns:
            out: (batch_size, seq_len, d_model)
            attn_weights: (batch_size, num_heads, seq_len, total_seq_len)
            new_kv_cache: Updated (key_cache, value_cache)
        """
        batch_size, seq_len, _ = x.shape 

        # 1. linear projections: (B, T, d_model) -> (B, T, d_model)
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 2. reshaping for multi-head: (B, T, d_model) -> (B, num_heads, T, d_k)
        q = q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. applying Rotary Position Embeddings (RoPE) to Q and K
        q, k = apply_rotary_emb(q, k, freqs_cis = freqs_cis)

        # 4. handling KV-Cache for auto-regressive generation
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim = 2)  # append along seq_len dim
            v = torch.cat([prev_v, v], dim = 2)

        new_kv_cache = (k, v)
        total_seq_len = k.size(2)

        # 3. handling causal masking
        # causal mask is required if sequence length > 1 during training/prefill
        mask = None 
        if is_causal and seq_len > 1:
            mask = causal_mask(seq_len, device = x.device)
        
        # 4. scaled dot-product attention
        # context shape: (B, num_heads, T, d_k)
        context, attn_weights = self.attention(q, k, v, mask = mask)

        # 5. concatenating heads: (B, num_heads, T, d_k) -> (B, T, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 6. final linear projection
        out = self.w_o(context)

        return out, attn_weights, new_kv_cache


# testing
if __name__ == "__main__":
    from .rope import precompute_freqs_cis

    B, T, d_model, H = 2, 5, 64, 8
    d_k = d_model // H
    x = torch.randn(B, T, d_model)

    # precomputing RoPE frequencies and slice for sequence
    freqs_cis = precompute_freqs_cis(dim=d_k, end=1024)
    freqs_cis_seq = freqs_cis[:T]
    
    mha = MultiHeadAttention(d_model=d_model, num_heads=H)
    out, weights, cache = mha(x, freqs_cis = freqs_cis_seq, is_causal=True)
    
    print("Input Shape:        ", x.shape)       # torch.Size([2, 5, 64])
    print("MultiHead Out Shape:", out.shape)     # torch.Size([2, 5, 64])
    print("Attn Weights Shape: ", weights.shape) # torch.Size([2, 8, 5, 5])
    print("KV Cache K shape: ", cache[0].shape)  # torch.Size([2, 8, 5, 8])