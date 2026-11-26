import torch
import torch.nn as nn
import torch.nn.functional as F



@torch.no_grad()
def generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    eos_token_id: int = None,
) -> torch.Tensor:
    """
    Generates new tokens auto-regressively using KV-caching and sampling strategies.
    
    Args:
        model: Transformer language model instance.
        input_ids: Prompt token IDs of shape (batch_size, seq_len).
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Controls randomness. Higher = more random, 0.0 = greedy argmax.
        top_k: Keeps only the top k highest probability tokens.
        top_p: Nucleus sampling threshold (retains tokens summing to cumulative probability top_p).
        eos_token_id: Optional end-of-sequence token ID to stop generation early.
        
    Returns:
        Tensor containing original input_ids plus newly generated tokens: (batch_size, seq_len + new_tokens)
    """
    model.eval()
    device = input_ids.device
    
    # prefilling phase: Process the full prompt once and initialize the KV-caches
    logits, kv_caches = model(input_ids)
    
    # extracting logits for the very last token in the prompt sequence
    next_token_logits = logits[:, -1, :]
    
    generated = [input_ids]
    
    for _ in range(max_new_tokens):
        # applying Temperature scaling
        if temperature == 0.0:
            next_token = torch.argmax(next_token_logits, dim = -1, keepdim = True)
        else:
            next_token_logits = next_token_logits / temperature
            
            # applying Top-k filtering
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                indices_to_remove = next_token_logits < values[:, [-1]]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # applying Top-p (Nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending = True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)
                
                # removing tokens with cumulative probability above the threshold p
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # shifting mask right to keep at least one token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False
                
                for batch_idx in range(next_token_logits.size(0)):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = float('-inf')
            
            # sample from the filtered probability distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
        generated.append(next_token)
        
        # checking for early stopping if EOS token is produced across all batch items
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
            
        # decoding phase: Pass ONLY the newly generated single token and existing KV-caches
        logits, kv_caches = model(next_token, kv_caches = kv_caches)
        next_token_logits = logits[:, -1, :]
        
    return torch.cat(generated, dim = 1)


if __name__ == "__main__":
    from models.transformer import Transformer

    vocab_size = 1000
    d_model = 64
    num_layers = 2
    num_heads = 4
    
    model = Transformer(
        vocab_size = vocab_size,
        d_model = d_model,
        num_layers = num_layers,
        num_heads = num_heads
    )
    
    # Prompt token IDs
    prompt_ids = torch.randint(0, vocab_size, (1, 3))
    
    # Generate 10 new tokens
    output_ids = generate(
        model = model,
        input_ids = prompt_ids,
        max_new_tokens = 10,
        temperature = 0.7,
        top_k = 50
    )

    print(output_ids)
    
    print("Prompt Shape:     ", prompt_ids.shape)  # torch.Size([1, 3])
    print("Generated Shape:  ", output_ids.shape)   # torch.Size([1, 13])