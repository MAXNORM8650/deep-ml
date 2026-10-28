import torch
import math
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    return torch.matmul(X, W_q), torch.matmul(X, W_k), torch.matmul(X, W_v)

import torch
import math

def masked_attention(Q, K, V, mask):
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])

    # if mask is not None:
    #     mask = mask.to(device=scores.device, dtype=torch.bool)
    #     scores = scores.masked_fill(~mask, float("-inf"))
    if mask is not None:
        scores = scores + mask
    attention = torch.softmax(scores, dim=-1)
    return attention @ V