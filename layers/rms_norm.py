import torch
import torch.nn as nn



class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Formula: RMSNorm(x) = (x / RMS(x)) * weight
    where RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Args:
            d_model: Dimensionality of the model embedding space.
            eps: Small value added to denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        # Learnable scaling parameter (gain) initialized to ones
        self.weight = nn.Parameter(torch.ones(d_model)) # special tensor for storing trainable parameters, it does not process data i.e no forward pass, 
                                                        # and only a single tensor object. we use torch.nn.Parameter to build custom operations that aren't
                                                        # available in standard layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Normalized tensor of shape (batch_size, seq_len, d_model)
        """
        # computing variance (mean of squares along the last dimension)
        variance = x.pow(2).mean(-1, keepdim = True)
        
        # normalizing by root mean square: x * 1 / sqrt(variance + eps)
        x_normed = x * torch.rsqrt(variance + self.eps)
        
        # scaling by the learnable gain parameter
        return x_normed * self.weight


if __name__ == "__main__":
    batch_size, seq_len, d_model = 2, 5, 64
    x = torch.randn(batch_size, seq_len, d_model)
    
    rms_norm = RMSNorm(d_model = d_model)
    out = rms_norm(x)
    
    print("Input Shape:  ", x.shape)   # torch.Size([2, 5, 64])
    print("Output Shape: ", out.shape)  # torch.Size([2, 5, 64])