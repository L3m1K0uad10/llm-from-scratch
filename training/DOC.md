Even though the underlying mathematical calculations (like gradients and weight updates) are handled by PyTorch, we wrote `training/loss.py` and `training/optimizer.py` to implement LLM-specific rules and configurations that standard PyTorch components don't know how to handle out of the box.


## loss.py

**What we implemented in `training/loss.py` (Causal Shifting)**

* What raw PyTorch does: `nn.CrossEntropyLoss()` takes a batch of predictions and a batch of targets and computes the error between them. It treats every token independently.

* What we implemented: **Next-token prediction shifting.**

    * In a causal LLM, token `t` must predict token `t+1`.

    * If we input sequence is `["The", "cat", "sat"]`, the model processes all three tokens at once.
    
    * We wrote the code to slice the logits (`logits[:, :-1, :]`) and targets (`targets[:, 1:]`) so that the model's prediction at position 0 ("The") is measured against position 1 ("cat"), position 1 ("cat") is measured against position 2 ("sat"), and so on. Without this custom wrapper, our LLM would try to predict the current token instead of the next one.

<br>

## optimizer.py

**What we implemented in training/optimizer.py (Parameter Grouping)**

* What raw PyTorch does: If we pass `torch.optim.AdamW(model.parameters())`, PyTorch applies weight decay (a penalty that shrinks weights to prevent overfitting) to every single parameter in our model equally.

* What we implemented: **Weight decay decoupling for LLMs.**

    * In modern LLMs, applying weight decay to 1D tensors—like the gain parameters in `RMSNorm` or model biases—hurts performance and model stability. Weight decay should only be applied to 2D weight matrices (like linear projections in attention and FFN blocks).

    * We wrote the loop that inspects every parameter by name and dimension (`param.ndim <= 1` or `"norm"` in name), splits them into two separate buckets, and tells AdamW: "Apply weight decay to the matrix weights, but set weight decay to 0.0 for all normalization weights."

<br>

## trainer.py
it ties the loss function, optimizer, and model together into a clean training loop that supports gradient clipping and batch updates.