from dataclasses import dataclass
import torch



def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class ModelConfig:
    # VOCAB_SIZE:
    # we can think of vocab size as the dictionary of the LLM. 
    # If a word or character is not in this dictionary, the model 
    # cannot understand it or output it

    # by setting this number (vocab_size), we are telling the tokenizer: 
    # "Scan our training data, find the (ex: 1000 or 32000, ...) most common words, 
    # sub-words, and characters, and throw away the rest. That is all this LLM is 
    # allowed to know."
    # hence It cannot use or understand any token outside of those


    # D_MODEL: MODEL DIMENSION (HIDDEN SIZE)
    # The width of the brain. The size of the vector (list of numbers) used to represent 
    # the meaning of every single token.


    # NUM_LAYERS:
    # The depth of the brain. How many Transformer blocks the text passes through sequentially.


    # NUM_HEADS: NUMBER OF ATTENTION HEADS
    # The multi-tasking capability. How many different contexts the model can focus 
    # on simultaneously during Multi-Head Attention.


    # MAX_SEQ_LEN: MAXIMUM SEQUENCE LENGTH (CONTEXT LENGTH)
    # The memory span. The absolute maximum number of tokens the model can process 
    # or generate in a single prompt/response.


    # DROPOUT_P: DROPOUT PROBABILITY
    # The anti-cheating mechanism. A decimal (e.g., 0.1 for 10%) representing the 
    # fraction of neural connections randomly turned off during training to prevent 
    # the model from just memorizing data.


    vocab_size: int = 1000
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 128
    dropout_p: float = 0.0

    # SOME COMPARATIVE ARCHITECTURAL BLUEPRINT
    # model, architecture type, d_model, num_layers, num_heads, max_seq_len, estimated vocab size
    # ChatGPT-4: MoE, ~12,288, 120 layers total, 96 heads, 128,000 tokens, 100,000+ tokens
    # Gemini 1.5 Pro: MoE, ~8,192, ~80+ layers, 64 heads, 1,000,000 to 2,000,000+ tokens, 256,000 tokens
    # Meta Llama 3.3: Dense, 8,192, 80 layers, 64 heads, 128,000 tokens, 128,256 tokens

    # DECENT SMALL LLM MODEL CONFIG
    # vocab_size: 32,000
    # d_model: 768
    # num_layers: 12
    # num_heads: 12
    # max_seq_len: 1,024 or 2,048
    # dropout_p: 0.1
    # Total Estimated Parameters: ~110 Million to ~130 Million (depending on the exact feed-forward network expansion factor).


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    device: str = get_default_device()
    checkpoint_path: str = "checkpoint.pt"

    # SOME COMPARATIVE ARCHITECTURAL BLUEPRINT
    # ChatGPT-4: MoE/Gemini 1.5 Pro: 
        # batch_size: uses Data Parallelism and Model Parallelism to achieve effective batch sizes of 1,024 to 2,048 tokens per GPU across thousands of GPUs.
        # epochs: 1 (They train on up to 15+ trillion tokens. Seeing the same data twice causes overfitting)
        # learning_rate: 1e-4 to 3e-4
        # weight_decay: standard 0.1
        # max_grad_norm: standard 1.0
        # device: Tens of thousands of interconnected NVIDIA H100/H200 or Google TPU clusters.
        # checkpoint_path: Distributed cloud storage clusters backing up terabytes of data hourly.
    # Meta Llama 3.3: Dense:
        # batch_size:  uses Data Parallelism and Model Parallelism to achieve effective batch sizes of 1,024 to 2,048 tokens per GPU across thousands of GPUs.
        # epochs: 1 (They train on up to 15+ trillion tokens. Seeing the same data twice causes overfitting)
        # epochs: 1 (They train on up to 15+ trillion tokens. Seeing the same data twice causes overfitting)
        # learning_rate: 1.5e-4 
        # weight_decay: standard 0.1
        # max_grad_norm: standard 1.0
        # device: Clusters of 24,000+ NVIDIA H100 GPUs running simultaneously.
        # checkpoint_path: Highly secure infrastructure buckets (saving a 70B model checkpoint takes ~140 GB).

    # No single GPU on Earth has enough VRAM memory to hold 4 million tokens at once. To achieve this, the engineering teams use two techniques:
    # 1. Data Parallelism: They divide those 512 sequences across thousands of GPUs (e.g., Llama 3 used a cluster of 24,000 H100 GPUs). Each individual GPU might only handle a local batch size of 1 or 2 sequences.
    # 2. Gradient Accumulation: The GPUs calculate the mathematical updates over multiple mini-steps without shifting the model's weights, micro-averaging them until they reach the 4-million-token target, and then update the master brain.

    # DECENT SMALL LLM MODEL CONFIG
    # batch_size: 16 or 32
    # epochs: 3 to 5 epochs
    # learning_rate: 3e-4 (0.0003)
    # weight_decay: 0.01 to 0.1
    # max_grad_norm: 1.0
    # device: "cuda" (Colab/Kaggle notebooks)
    # checkpoint_path: "/content/drive/MyDrive/llm_checkpoints/"
        # we map this directly to a folder inside our mounted 
        # Google Drive so timeouts don't erase our progress.


    # For the small model configuration we set up (batch_size = 32, max_seq_len = 1024, and grad_accum_steps = 4):
    # 32 * 1024 * 4 = 131,072 tokens per step
    # This means our small model will have to learn from 131,072 tokens every single time it updates its weights, 
    # which is perfectly proportional to its ~120M size!