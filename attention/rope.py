import torch 
import torch.nn as nn



def precompute_freqs_cis(
        dim: int,
        end: int,
        theta: float = 10000.0
) -> torch.Tensor:
    """
    Precomputes complex frequencies for Rotary Position Embeddings (RoPE).
    
    Args:
        dim: Head dimension (d_k). Must be even.
        end: Maximum sequence length.
        theta: Frequency scaling factor (default: 10000.0).
        
    Returns:
        Complex tensor of shape (end, dim // 2) representing e^(i * theta).
    """
    assert dim % 2 == 0, "Head dimension must be even for RoPE."

    # computing base frequencies: theta_i = 1.0 / (theta ^ (2i / dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # sequence position indices [0, 1, ..., end - 1]
    t = torch.arange(end, dtype = torch.float32)

    # outer product to get angles for every (position, freq_pair): shape (end, dim // 2)
    freqs = torch.outer(t, freqs)

    # converting angles to complex polar form: cos(angle) + i * sin(angle)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Position Embeddings to Query and Key tensors.
    
    Args:
        xq: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
        xk: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
        freqs_cis: Complex frequencies of shape (seq_len, d_k // 2)
        
    Returns:
        Rotated Query and Key tensors with matching input shapes.
    """
    # 1. reshaping real tensors into complex pairs across the head dimension
    # (B, H, T, d_k) -> (B, H, T, d_k // 2, 2) -> complex view (B, H, T, d_k // 2)
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 2. reshaping freqs_cis for broadcasting: (1, 1, seq_len, d_k // 2)
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    
    # 3. rotating via complex multiplication
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
    
    return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__ == "__main__":
    B, H, T, d_k = 2, 4, 8, 16
    q = torch.randn(B, H, T, d_k)
    k = torch.randn(B, H, T, d_k)
    
    freqs_cis = precompute_freqs_cis(dim = d_k, end = 1024)
    freqs_cis_seq = freqs_cis[:T]  # slice for current sequence length
    
    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis_seq)
    
    print("Rotated Q shape:", q_rot.shape)  # torch.Size([2, 4, 8, 16])
    print("Rotated K shape:", k_rot.shape)  # torch.Size([2, 4, 8, 16])