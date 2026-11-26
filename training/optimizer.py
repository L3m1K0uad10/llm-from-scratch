import torch
import torch.nn as nn



def get_optimizer(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95
) -> torch.optim.AdamW:
    """
    Configures the AdamW optimizer with weight decay exclusion for norms and biases.
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # excluding 1D tensors (RMSNorm weights) and biases from weight decay
        if param.ndim <= 1 or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]
    
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr = learning_rate,
        betas = (beta1, beta2)
    )
    return optimizer