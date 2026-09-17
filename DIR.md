
```
llm/
├── __init__.py
│
├── attention/
│   ├── __init__.py
│   ├── scaled_dot_product.py
│   ├── multi_head.py           # Includes RoPE & KV-Cache integration
│   ├── causal.py
|   └── rope.py                 # Rotary Position Embeddings
│
├── embeddings/
│   ├── __init__.py
│   └── token.py                # Token Embeddings
│
├── layers/
│   ├── __init__.py
│   ├── transformer_block.py    # Combines MHA + FeedForward + Norms
│   ├── feed_forward.py         # SwiGLU / Standard MLP
│   └── rms_norm.py             # RMSNorm (modern replacement for LayerNorm)
│
├── models/
│   ├── __init__.py
│   ├── transformer.py          # Complete Causal Decoder Transformer
│   └── generation.py           # Generation loop, KV-cache management, sampling
│
├── tokenizer/
│   ├── __init__.py
│   └── tokenizer.py            # Your custom BPE Tokenizer
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── loss.py                 # CrossEntropyLoss wrapper
│   └── optimizer.py
│
├── config.py                   # Model and Training hyperparameters           
├── generate.py                 # CLI entrypoint to generate text           
└── train.py                    # CLI entrypoint to train the model
```