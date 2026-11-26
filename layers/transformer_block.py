import torch
import torch.nn as nn

from layers.rms_norm import RMSNorm
from layers.feed_forward import FeedForward
from attention.multi_head import MultiHeadAttention



class TransformerBlock(nn.Module):
    """
    Decoder Transformer Block consisting of:
    1. Pre-RMSNorm -> Multi-Head Self-Attention -> Residual Connection
    2. Pre-RMSNorm -> Feed-Forward Network (SwiGLU) -> Residual Connection
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attention = MultiHeadAttention(d_model = d_model, num_heads = num_heads, dropout_p = dropout_p)
        
        self.ffn_norm = RMSNorm(d_model)
        self.feed_forward = FeedForward(d_model = d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] = None,
        is_causal: bool = True
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            freqs_cis: Precomputed RoPE frequencies sliced to current sequence positions
            kv_cache: Optional tuple of (cached_k, cached_v) for fast inference
            is_causal: Whether to apply causal mask
            
        Returns:
            out: Transformed tensor of shape (batch_size, seq_len, d_model)
            new_kv_cache: Updated (key_cache, value_cache) tuple
        """
        # 1. implementing the first block, Self-Attention Sub-Layer (Pre-Norm + Residual)
        normed_x = self.attn_norm(x)
        attn_out, _, new_kv_cache = self.attention(
            normed_x, 
            freqs_cis = freqs_cis, 
            kv_cache = kv_cache, 
            is_causal = is_causal
        )
        x = x + attn_out
        
        # 2. implementing the second block, Feed-Forward Sub-Layer (Pre-Norm + Residual)
        normed_x = self.ffn_norm(x)
        ffn_out = self.feed_forward(normed_x)
        x = x + ffn_out
        
        return x, new_kv_cache


if __name__ == "__main__":
    from attention.rope import precompute_freqs_cis

    B, T, d_model, H = 2, 5, 64, 8
    d_k = d_model // H
    x = torch.randn(B, T, d_model)
    
    # Precompute RoPE frequencies
    freqs_cis = precompute_freqs_cis(dim = d_k, end = 1024)[:T]
    
    block = TransformerBlock(d_model = d_model, num_heads = H)
    out, cache = block(x, freqs_cis = freqs_cis)
    
    print("Transformer Block Input Shape: ", x.shape)       # torch.Size([2, 5, 64])
    print("Transformer Block Output Shape:", out.shape)      # torch.Size([2, 5, 64])
    print("KV Cache K Shape:             ", cache[0].shape)  # torch.Size([2, 8, 5, 8])