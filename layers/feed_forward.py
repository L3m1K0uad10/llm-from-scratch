import torch
import torch.nn as nn
import torch.nn.functional as F



class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network (FFN).
    Formula: down_proj(SiLU(gate_proj(x)) * up_proj(x))
    """
    def __init__(self, d_model: int, hidden_dim: int = None):
        """
        Args:
            d_model: Dimensionality of the model embedding space.
            hidden_dim: Intermediate hidden dimension. If None, uses the standard
                        Llama expansion factor (~8/3 * d_model) rounded to a multiple of 256.
        """
        super().__init__()
        if hidden_dim is None:
            # Standard Llama convention: 8/3 * d_model, rounded to nearest multiple of 256
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = ((hidden_dim + 255) // 256) * 256

        # Gate and Up projections run in parallel
        self.w_gate = nn.Linear(d_model, hidden_dim, bias = False)
        self.w_up = nn.Linear(d_model, hidden_dim, bias = False)
        
        # Down projection maps back to d_model
        self.w_down = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # applying SiLU activation to the gate projection and multiply element-wise with up projection
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        intermediate = gate * up
        
        # projecting back down to d_model
        return self.w_down(intermediate)


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 5, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    ffn = FeedForward(d_model = d_model)
    out = ffn(x)
    
    print("Input Shape:  ", x.shape)   # torch.Size([2, 5, 64])
    print("Output Shape: ", out.shape)  # torch.Size([2, 5, 64])