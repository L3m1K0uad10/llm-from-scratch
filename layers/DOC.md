
```
├── layers/
│   ├── __init__.py
│   ├── rms_norm.py
│   ├── feed_forward.py
│   └── transformer_block.py
```


## rms_norm.py: 
Root Mean Square Normalization (RMSNorm) is the normalization standard used by LLMs(such as Llama, Mistral, and Qwen). Unlike traditional LayerNorm, RMSNorm drops the mean-centering step and only normalizes by the root mean square, which speeds up computation while maintaining model stability.

**RMSNorm (Root Mean Square Normalization) is a feature-scaling technique used in deep learning to stabilize training.**

To understand what it means exactly, it helps to compare it to traditional LayerNorm. Traditional LayerNorm calculates both the mean (average) and the variance (spread) of the activations, shifts the data by subtracting the mean, and scales it by the variance.

RMSNorm simplifies this by assuming the mean is already close to zero. Instead of calculating and subtracting the mean, it skips that step entirely and only scales the data using the Root Mean Square (RMS). Because it does fewer mathematical operations, it is frequently 10% to 50% faster than LayerNorm while achieving the exact same model accuracy.


A Mathematical Example:

Let's say a hidden layer in a language model outputs a small vector of 4 activation values for a specific token:

`x =[2,-1,3,0]`

1. calculating the Square of each value

First, we square all the numbers in the vector to eliminate negative values:
```
2^2 = 4, (-1)^2 = 1, 3^2 = 9, 0^2 = 0
x^2 = [4,1,9,0]
```

2. calculating the Mean (Average) of those squares

We sum the squared values and divide by the number of elements `(n = 4)`:

`Mean = (4+1+9+0)/4 = 14/4 = 3.5`

3. taking the Root (Square Root)

This gives us the Root Mean Square (RMS) value:

`RMS = √3.5 ≈ 1.8708`

4. normalizing the original vector

Divide each original value in `x` by the RMS value to get the scaled output:
• 2 / 1.8708 ≈ 1.069
• -1 / 1.8708 ≈ -0.535
• 3 / 1.8708 ≈ 1.604
• 0 / 1.8708 ≈ 0.000

Normalized Vector: [1.069, -0.535, 1.604, 0.000]

_(Note: In practice, a tiny numbers like 10^-6 is added to the variance before the square root to prevent dividing by zero, and the final output is multiplied by a learnable scaling parameter γ, but this is the core calculation)._


**WHY EVEN NORMALIZATION?**

We keep the vector the same shape because the network's layers expect a specific size, but we normalize the numbers inside it to prevent the neural network from exploding or dying during training.

Without normalization, deep language models like Llama or Mistral completely fail to learn. Here is exactly why we cannot use the original vector:


1. The "Exploding Gradients" Problem

Deep LLMs stack dozens of layers on top of each other (Llama 3 has up to 126 layers). In a neural network, layers perform matrix multiplication.

If the numbers in our vector are slightly larger than 1 (like the `3` in our previous example), and they get multiplied by weights layer after layer after layer, those numbers grow exponentially. By the time the vector reaches layer 80, the numbers become so massive that the computer runs out of memory (`NaN` or Infinity errors), destroying the model. **Normalization keeps the scale of the numbers constrained at every single layer.**


2. Internal Covariate Shift

As an LLM trains, the weights in the early layers change constantly. If we don't normalize, a small change in Layer 2 might cause the output of Layer 3 to wildly shift its scale. Layer 4 then has to completely relearn how to handle this new scale, creating a chaotic domino effect.

Normalization ensures that no matter how much the weights change, the inputs to the next layer always look statistically similar (they share a steady scale).

3. Equalizing Feature Importance

Imagine a vector representing two tokens:
• Vector A: [100, 200, 150]
• Vector B: [0.1, 0.2, 0.15]

Both vectors share the exact same internal pattern, but Vector A's numbers are 1,000 times larger. Without normalization, the model's optimization math (gradient descent) would obsess over Vector A because its large numbers generate massive gradients, completely ignoring the subtle signals in Vector B. Normalization puts both vectors on a level playing field.

<br>

## feed_forward.py:
improves the model's representation capacity and training dynamics.

Modern LLMs (such as Llama, Mistral, and PaLM) use SwiGLU (Swish-Gated Linear Unit) instead of traditional MLPs.

Instead of passing data through a single linear layer followed by a fixed activation function (like ReLU or GELU), SwiGLU splits the input into two parallel learned paths and multiplies them together element-wise. One of those paths acts as a dynamic "gate" to regulate the information passing through the other.

`SwiGLU(x) = (Swish(xW_1) ⊙ xW_2)W_3`

<br>

## transformer_block.py
combines `RMSNorm`, `MultiHeadAttention` (with RoPE and KV-Cache support), and the SwiGLU `FeedForward` network into a single standard decoder block using Pre-Normalization and Residual Connections.