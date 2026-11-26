

## transformer.py
it ties together everything we've built so far into a complete, end-to-end Causal Decoder-Only Transformer Language Model:

1. Token Embedding lookup with variance scaling.

2. Stacked `TransformerBlock` layers (using `nn.ModuleList`).

3. Precomputed RoPE frequency buffers registered safely inside the module state.

4. Final `RMSNorm` normalization.

5. Output Language Model (`lm_head`) projection mapping hidden states back to vocabulary logits.


**A Causal Decoder-Only Transformer Language model** is a neural network architecture that generates text and code by predicting the next token (such as a word, subword, or character) in a sequence from left to right

important keyword understanding for later reference:
we use `nn.Sequential` when our data flows in a straight line from one layer to the next. we use `nn.ModuleList` when that straight line breaks.

Here are the three clearest signs that you need `nn.ModuleList`:

1. If we are building a ResNet or U-Net where an early input needs to skip ahead and blend with a later layer, `nn.Sequential` cannot handle it because it only passes data strictly to the immediate next layer.
    • With ModuleList: You can manually grab `layer[i](x)` and add it back to x later.

2. If we want to pass data through a variable number of layers dynamically, or apply a specific operation (like saving intermediate activations) inside a loop, use `nn.ModuleList`.

3. If we are building an architecture where parallel branches process the same input at the same time (like an Inception module or certain Multi-Head Attention components), `nn.Sequential` will fail because it forces a cascade chain

<br>

## generation.py
it implements efficient auto-regressive text generation using KV-caching and flexible sampling strategies (Temperature scaling, Top-k, and Top-p / Nucleus sampling).