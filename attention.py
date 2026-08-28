import numpy as np



d = 4 # embedding dimension
d_k = 4 # 1 attention head



# example of sequence and static token embeddings (X)
tokens = ["The", "cat", "sat"]

X = np.array([[0.1, 0.8, 0.0, 0.5],  # token 0: "The"
              [0.9, 0.2, 0.7, 0.1],  # token 1: "cat"
              [0.3, 0.6, 0.1, 0.8]]) # token 2: "sat"

"""
print(f"data type of X: {type(X)}")
print(f"data type of X values: {X.dtype}")
print(f"token 0: {X[0]}")
print(f"token 1: {X[1]}")
print(f"token 2: {X[2]}")
"""



# linear projections (Q, K, V)
# X (3 x 4) dimension --- W_q, W_k, W_v (4 x 4) dimension

## weight matrices
W_q = np.identity(4)
W_k = np.identity(4)
W_v = np.identity(4) * 10

"""
print(W_q)
print(W_k)
print(W_v)
"""

# query matrix Q = X . W_q
Q = X @ W_q 

# key matrix K = X . W_k
K = X @ W_k 

# value matrix V = X . W_v
V = X @ W_v 

"""
print(Q)
print(K)
print(V)
"""



# computing raw attention score S = Q . K^T
S = Q @ K.T

"""
print(S)
"""



# scaling the scores √d_k
sqrt_dk = np.sqrt(d_k) # d_k = 4 => √d_k = √4 = 2. divide every value in S by 2
S_scaled = S / sqrt_dk

"""
print(S_scaled)
"""



# applying causal masking
# to prevent token 2 ("sat") or earlier tokens from looking into future tokens,
# we set all entries above the main diagonal to -∞

masked_S = S_scaled.copy()
masked_S[np.triu_indices_from(masked_S, k = 1)] = -np.inf 

"""
print(masked_S)
"""



# softmax across rows (attention weights A)
#                 e^x_i-max(x)
# softmax(x_i) = --------------  controls instability of np.exp() value being too large for floating point number
#                ∑ e^x_j-max(x)
#
# final attention matrix A

def softmax(array:np.ndarray, axis = -1) -> np.ndarray:
    x = np.exp(array - np.max(array, axis = axis, keepdims = True))

    return x / np.sum(x, axis = axis, keepdims = True)   

A = softmax(masked_S)

"""
print(A)
"""



# output context vectors O = A . V
# final contextual output O
O = A @ V 

print(O)




# TODO: Refactor attention into a class.
# - Create an Attention class.
# - Move W_q, W_k, W_v into __init__ as instance attributes.
# - Initialize the weight matrices randomly.
# - Move the attention calculation into a forward() method.
#
# TODO: Add batch dimension support.
# Current input shape: (seq_len, embedding_dim)
# Target input shape:  (batch_size, seq_len, embedding_dim)
# - Update Q, K, V projections to work with 3D tensors.
# - Update matrix multiplication for batched inputs.
# - Verify output shape is (batch_size, seq_len, embedding_dim).
#
# TODO: Implement multi-head attention.
# - Split the embedding dimension into multiple attention heads.
# - Reshape Q, K, V to separate the head dimension.
# - Compute attention independently for each head.
# - Concatenate the heads back together.
# - Add a final output projection.
