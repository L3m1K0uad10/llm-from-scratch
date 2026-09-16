import torch 



def causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """  
    Creates a boolean causal mask of shape (1, 1, seq_len, seq_len).
    True values represent positions that MUST be masked out (-inf)

    Args: 
        seq_len: length of the input sequence
        device: PyTorch device (cpu/cuda)

    Returns:
        Boolean Tensor where upper triangular elements (above main diagonal) are True.
    """
    # torch.triu with diagonal = 1 sets lower triangular + main diagonal to 0 (False)
    # and upper triangular positions to 1 (True)
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype = torch.bool, device = device), diagonal = 1)

    # unsqueeze twice for broadcasting accross batch_size and num_heads dimensions
    # final shape: (1, 1, seq_len, seq_len)
    return mask.unsqueeze(0).unsqueeze(0)


# testing
if __name__ == "__main__":
    seq_len = 4
    mask = causal_mask(seq_len)
    print("Causal Mask Shape:", mask.shape)
    print("Mask Tensor:\n", mask.squeeze())