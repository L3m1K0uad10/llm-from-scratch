from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

from .config import ModelConfig, TrainConfig
from .models.transformer import Transformer
from .tokenizer.tokenizer import Tokenizer
from .training.trainer import Trainer


class TokenBlockDataset(Dataset):
    """Creates overlapping (input_ids, target_ids) blocks of length seq_len."""

    def __init__(self, token_ids: list[int], seq_len: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Need at least seq_len tokens so shifting by 1 still leaves valid targets
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.seq_len]
        # CausalLMLoss shifts logits/targets internally, so input and target share the window
        return chunk, chunk.clone()


def load_corpus() -> list[str]:
    sample_corpus = [
        "The cat sat on the mat.",
        "The dog sat on the rug.",
        "Cats and dogs like the mat!",
        "Does a cat like a rug?",
        "A small cat sleeps on the warm mat.",
        "The big dog runs across the rug.",
    ]
    return sample_corpus * 40


def main() -> None:
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()

    corpus = load_corpus()

    tokenizer = Tokenizer()
    tokenizer.train(corpus, target_vocab_size=model_cfg.vocab_size)
    # Keep vocab_size aligned with actual trained vocabulary
    model_cfg.vocab_size = tokenizer.vocab_size

    joined_text = "\n".join(corpus)
    token_ids = tokenizer.encode(joined_text)

    seq_len = min(model_cfg.max_seq_len, max(2, len(token_ids) - 1))
    dataset = TokenBlockDataset(token_ids=token_ids, seq_len=seq_len)
    train_loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
    )

    model = Transformer(
        vocab_size=model_cfg.vocab_size,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        max_seq_len=model_cfg.max_seq_len,
        dropout_p=model_cfg.dropout_p,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        device=train_cfg.device,
    )

    print(f"Training on device: {train_cfg.device}")
    for epoch in range(1, train_cfg.epochs + 1):
        avg_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch} complete | Average Loss: {avg_loss:.4f}")

    # Store merges ordered by rank so BPE priority order is preserved in the checkpoint
    ordered_merges = [
        list(pair)
        for pair, _ in sorted(tokenizer.merge_ranks.items(), key=lambda item: item[1])
    ]

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_cfg.__dict__,
        "vocab": tokenizer.vocab,
        "merges": ordered_merges,
    }
    Path(train_cfg.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, train_cfg.checkpoint_path)
    print(f"Saved checkpoint to {train_cfg.checkpoint_path}")


if __name__ == "__main__":
    main()