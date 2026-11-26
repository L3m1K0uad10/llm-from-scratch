
## config.py: 
Using `@dataclass` keeps model hyperparameters and training settings typed, structured, and easy to swap in one place


## train.py:
This script loads `ModelConfig` and `TrainConfig`, builds a sliding-window `Dataset` and `DataLoader` from sample text (or a text file), instantiates `Transformer` and `Trainer`, runs the training loop, and saves a model checkpoint.

Note on `encode / decode`: right now `tokenizer/tokenizer.py` implemented `train()`, `save()`, and `load()`. In `train.py`, for now we included a small helper `encode_text()` that maps tokens via `tokenizer.vocab`, once we add a full `.encode()` method to `Tokenizer`, WE COULD REPLACE THAT HELPER DIRECTLY WITH `tokenizer.encode(...)`.


## generate.py:
It loads the checkpoint saved by `train.py` (including `model_state_dict`, `model_config`, and `vocab`), rebuilds the `Tokenizer` and `Transformer`, encodes a CLI prompt, calls `generate()` with KV-caching and sampling options, and decodes the generated token IDs back into text.