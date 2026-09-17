import torch 
import torch.nn as nn 



class TokenEmbedding(nn.Module):
    """
    Token Embedding layer that maps discrete token IDs to dense vectors.
    Scales embeddings by sqrt(d_model) to stabilize variance before Attention layers.
    """
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Total number of unique tokens in vocabulary.
            d_model: Dimensionality of the model embedding space.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model 

        # standard PyTorch lookup table (vocab_size x d_model)
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Integer tensor of token IDs with shape (batch_size, seq_len)
            
        Returns:
            Dense float tensor with shape (batch_size, seq_len, d_model)
        """
        # multiplying by sqrt(d_model) following standard Transformer initialization conventions
        return self.embedding(x) * (self.d_model ** 0.5)
    

if __name__ == "__main__":
    vocab_size = 50000  # e.g., standard BPE vocabulary size
    d_model = 64
    batch_size = 2
    seq_len = 5

    # simulated batch of token IDs
    input_ids = torch.tensor([
        [101, 2023, 2003, 1037, 3899],
        [101, 3899, 2003, 2023, 102]
    ])

    token_emb = TokenEmbedding(vocab_size = vocab_size, d_model = d_model)
    embeddings = token_emb(input_ids)

    print("Input IDs shape:  ", input_ids.shape)   # torch.Size([2, 5])
    print("Embedding shape:  ", embeddings.shape)  # torch.Size([2, 5, 64])
    #print(embeddings)
