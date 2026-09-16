llm/
├── __init__.py
│
├── attention/
│   ├── __init__.py
│   ├── scaled_dot_product.py
│   ├── multi_head.py
│   └── causal.py
│
├── embeddings/
│   ├── __init__.py
│   ├── token.py
│   └── positional.py
│
├── layers/
│   ├── __init__.py
│   ├── transformer_block.py
│   ├── feed_forward.py
│   ├── layer_norm.py
│   └── dropout.py
│
├── models/
│   ├── __init__.py
│   └── transformer.py
│
├── tokenizer/
│   ├── __init__.py
│   └── tokenizer.py
│
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   ├── loss.py
│   └── optimizer.py
│
├── config.py
└── train.py