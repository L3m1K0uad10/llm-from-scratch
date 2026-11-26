import torch
import torch.nn as nn



class CausalLMLoss(nn.Module):
    """
    Cross-entropy loss wrapper for Causal Language Modeling.
    Shifts logits and targets to evaluate next-token prediction.
    """
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            targets: LongTensor of shape (batch_size, seq_len) containing token IDs
            
        Returns:
            Scalar cross-entropy loss tensor.
        """
        # shifting logits and targets so token at t predicts t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        
        # flattening tensors for CrossEntropyLoss: (batch_size * (seq_len - 1), vocab_size)
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1)
        )
        return loss