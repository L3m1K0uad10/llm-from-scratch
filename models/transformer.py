import torch
import torch.nn as nn

from embeddings.token import TokenEmbedding
from layers.transformer_block import TransformerBlock
from layers.rms_norm import RMSNorm
from attention.rope import precompute_freqs_cis



class Transformer(nn.Module):
    """
    Complete Causal Decoder-Only Transformer Language Model.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout_p: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # token Embeddings
        self.token_embedding = TokenEmbedding(vocab_size = vocab_size, d_model = d_model)
        
        # stacked Transformer Blocks
        # nn.ModuleList vs. nn.Sequential
        # nn.ModuleList: can be dynamic, non-sequential, or conditional. Dynamic architectures, layer iteration, skip/residual connections, or tracking intermediate activations.
        # nn.Sequential: strictly linear and sequential. Simple cascading blocks where input flows linearly through all layers.
        self.layers = nn.ModuleList([
            TransformerBlock(d_model = d_model, num_heads = num_heads, dropout_p = dropout_p)
            for _ in range(num_layers)
        ])
        
        # final RMSNorm
        self.norm = RMSNorm(d_model)
        
        # output Language Model Head (projects hidden states -> vocabulary logits)
        self.lm_head = nn.Linear(d_model, vocab_size, bias = False)
        
        # precompute RoPE frequencies buffer (not a learnable parameter, stored with model state)
        d_k = d_model // num_heads
        freqs_cis = precompute_freqs_cis(dim = d_k, end = max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis, persistent = False)

    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = None,
        is_causal: bool = True
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids: Token ID tensor of shape (batch_size, seq_len)
            kv_caches: Optional list of length num_layers containing (k, v) caches per block
            is_causal: Whether to apply causal masking
            
        Returns:
            logits: Output vocabulary logits of shape (batch_size, seq_len, vocab_size)
            new_kv_caches: Updated list of KV caches per layer
        """
        batch_size, seq_len = input_ids.shape
        
        # handling KV cache initialization if not provided
        if kv_caches is None:
            kv_caches = [None] * self.num_layers
            
        # determining start position for inference or prefill
        past_len = kv_caches[0][0].size(2) if kv_caches[0] is not None else 0
        
        # slicing precomputed RoPE frequencies for the current sequence window
        freqs_cis_seq = self.freqs_cis[past_len : past_len + seq_len]
        
        # token Embedding lookup
        h = self.token_embedding(input_ids)
        
        # passing through stacked Transformer blocks
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            h, new_cache = layer(
                h,
                freqs_cis = freqs_cis_seq,
                kv_cache = kv_caches[i],
                is_causal = is_causal
            )
            new_kv_caches.append(new_cache)
            
        # final Normalization
        h = self.norm(h)
        
        # LM Head projection to vocabulary logits
        logits = self.lm_head(h)
        
        return logits, new_kv_caches


if __name__ == "__main__":
    vocab_size = 1000
    d_model = 64
    num_layers = 2
    num_heads = 4
    batch_size = 2
    seq_len = 5

    # Simulated input token IDs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    model = Transformer(
        vocab_size = vocab_size,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads
    )
    
    logits, caches = model(input_ids)

    print("Input Token IDs Shape:", input_ids.shape)  # torch.Size([2, 5])
    print("Output Logits Shape:  ", logits.shape)     # torch.Size([2, 5, 1000])
    print("Number of KV Caches:  ", len(caches))      # 2 (one per layer)
    print("Layer 0 Cache K Shape:", caches[0][0].shape) # torch.Size([2, 4, 5, 16])