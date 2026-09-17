
```
├── attention/
│   ├── __init__.py
│   ├── scaled_dot_product.py
│   ├── multi_head.py
│   ├── rope.py
│   └── causal.py
```


## scaled_dot_product.py: 
implement the core attention operation.
```
Q @ Kᵀ
   ↓
similarity scores
   ↓
divide by √d_k
   ↓
apply causal mask
   ↓
softmax
   ↓
multiply by V
   ↓
attention output
```


## causal.py: 
handle the causal mask — the mechanism that prevents a GPT-style LLM from looking at future tokens.


## multi_head.py:
build Multi-Head Self-Attention — the part that takes the basic attention operation from scaled_dot_product.py and runs several attention "heads" in parallel.

## rope.py
encodes relative position by rotating Q and K vectors in complex pairs rather than adding static positional vectors to input embeddings.