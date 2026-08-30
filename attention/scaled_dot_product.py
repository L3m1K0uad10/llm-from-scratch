import math
import torch
import torch.nn as nn
import torch.nn.functional as F



class ScaledDotProductAttention(nn.Module):
    """
    Computes Scaled Dot-Product Attention:
        Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + mask) @ V
    """
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout_p = dropout_p 
        self.dropout = nn.Dropout(dropout_p) 

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: Tensor of shape (batch_size, num_heads, seq_len_q, d_k)
            key:   Tensor of shape (batch_size, num_heads, seq_len_k, d_k)
            value: Tensor of shape (batch_size, num_heads, seq_len_k, d_v)
            mask:  Optional Tensor broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
                   Contains 0 (or False) for allowed tokens, and -inf (or True/boolean mask) for masked tokens.

        Returns:
            output: Context-enriched output tensor of shape (batch_size, num_heads, seq_len_q, d_v)
            attn_weights: Softmax attention scores of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """

        # obtaining dimension size d_k for scaling
        d_k = query.size(-1) 

        # Q @ K^T -> similarity scores
        # query shape: (..., seq_len_q, d_k)
        # key.transpose(-2, -1) shape: (..., d_k, seq_len_k)
        # scores shape: (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) # we don't transpose batch_size, num_heads only seq_len_k, d_k

        # dividing by sqrt(d_k)
        scores = scores / math.sqrt(d_k) 

        # applying mask (if provided)
        if mask is not None:
            #if mask is boolean (True where masked, False where valid):
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float('-inf')) # mask is a boolean from torch.triu(torch.ones(n, n), diagonal=1).bool() for example
            else:
                # if mask contains -inf values directly:
                scores = scores + mask

        # softmax along the last dimension (seq_len_k)
        attn_weights = F.softmax(scores, dim = -1)

        # applying dropout to attention weights (standard during LLM training)
        attn_weights = self.dropout(attn_weights)

        # multiplying by V -> attention output
        # attn_weights shape: (..., seq_len_q, seq_len_k)
        # value shape: (..., seq_len_k, d_v)
        # output shape: (batch_size, num_heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights, value)

        return output, attn_weights
    

# basic testing
if __name__ == "__main__":
    # test batch parameters: Batch = 2, Heads = 4, Seq_len = 3, d_k = 8
    B, H, T, d_k = 2, 4, 3, 8

    Q = torch.randn(B, H, T, d_k)
    K = torch.randn(B, H, T, d_k)
    V = torch.randn(B, H, T, d_k)

    attention_layer = ScaledDotProductAttention(dropout_p = 0.0)
    out, weights = attention_layer(Q, K, V)

    print("Output Shape: ", out.shape)      # torch.Size([2, 4, 3, 8])
    print("Weights Shape:", weights.shape)  # torch.Size([2, 4, 3, 3])