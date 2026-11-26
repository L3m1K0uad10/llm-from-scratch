import argparse
import torch

from .config import TrainConfig, get_default_device
from .models.generation import generate
from .models.transformer import Transformer
from .tokenizer.tokenizer import Tokenizer



def build_tokenizer_from_checkpoint(checkpoint: dict) -> Tokenizer:
    """Reconstructs a Tokenizer with both vocab and rank-ordered BPE merges."""
    
    tokenizer = Tokenizer()
    tokenizer.vocab = {int(k): v for k, v in checkpoint["vocab"].items()}
    raw_merges = checkpoint.get("merges", [])
    tokenizer.merge_ranks = {
        (pair[0], pair[1]): idx for idx, pair in enumerate(raw_merges)
    }
    tokenizer._rebuild_lookups()
    return tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Generate text from a trained LLM checkpoint.")

    parser.add_argument(
        "--prompt",
        type = str,
        required = True,
        help = "Input text prompt to continue.",
    )

    parser.add_argument(
        "--checkpoint",
        type = str,
        default = TrainConfig.checkpoint_path,
        help = "Path to the saved model checkpoint.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type = int,
        default = 30,
        help = "Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type = float,
        default = 0.8,
        help = "Sampling temperature (0.0 for greedy argmax).",
    )

    parser.add_argument(
        "--top-k",
        type = int,
        default = 20,
        help = "Keep only the top k highest-probability tokens during sampling.",
    )

    parser.add_argument(
        "--top-p",
        type = float,
        default = None,
        help = "Optional nucleus sampling probability threshold (between 0.0 and 1.0).",
    )

    parser.add_argument(
        "--device",
        type = str,
        default = None,
        help = "Target device (cuda, mps, or cpu). Defaults to auto-detected device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or get_default_device()

    checkpoint = torch.load(args.checkpoint, map_location = device)
    model_cfg = checkpoint["model_config"]
    vocab = checkpoint["vocab"]

    tokenizer = build_tokenizer_from_checkpoint(vocab)

    model = Transformer(
        vocab_size = model_cfg["vocab_size"],
        d_model = model_cfg["d_model"],
        num_layers = model_cfg["num_layers"],
        num_heads = model_cfg["num_heads"],
        max_seq_len = model_cfg.get("max_seq_len", 2048),
        dropout_p = model_cfg.get("dropout_p", 0.0),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt produced no valid token IDs for the checkpoint vocabulary.")

    input_ids = torch.tensor([prompt_ids], dtype = torch.long, device = device)

    output_ids = generate(
        model = model,
        input_ids = input_ids,
        max_new_tokens = args.max_new_tokens,
        temperature = args.temperature,
        top_k = args.top_k,
        top_p = args.top_p,
    )

    generated_tokens = output_ids[0].tolist()
    decoded_text = tokenizer.decode(generated_tokens)

    print("Prompt:   ", args.prompt)
    print("Generated:", decoded_text)


if __name__ == "__main__":
    main()