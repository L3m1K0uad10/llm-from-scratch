import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .loss import CausalLMLoss
from .optimizer import get_optimizer



class Trainer:
    """
    Training pipeline manager for the Transformer language model.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_grad_norm: float = 1.0,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        self.loss_fn = CausalLMLoss()
        self.optimizer = get_optimizer(
            model = self.model,
            learning_rate = learning_rate,
            weight_decay = weight_decay
        )

    def train_epoch(self, epoch: int) -> float:
        """Runs a single training epoch across all data batches."""
        
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # forward pass
            logits, _ = self.model(inputs)
            loss = self.loss_fn(logits, targets)
            
            # backward pass & optimization
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # cliping gradients to prevent explosion
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # updating weights
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item():.4f}")
                
        return total_loss / len(self.train_loader)