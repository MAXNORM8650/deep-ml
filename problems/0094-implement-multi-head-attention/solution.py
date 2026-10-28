import torch
import torch.nn.functional as F
from typing import Tuple
import math
def compute_qkv(X: torch.Tensor, W_q: torch.Tensor, W_k: torch.Tensor, W_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute Query, Key, and Value matrices.

    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)

    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here

    return X@W_q, X@W_k, X@W_v

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product self-attention.

    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)

    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    score =(Q@K.transpose(-2,-1))/math.sqrt(Q.shape[-1])
    softmax_score = torch.nn.functional.softmax(score, dim=-1)
    return softmax_score@V

def multi_head_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, n_heads: int) -> torch.Tensor:
    """
    Compute multi-head attention.

    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads

    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    if d_model%n_heads!=0:
        raise ValueError("d_model%n_heads must be 0")
    Q = Q.view(seq_len,n_heads,int(d_model/n_heads))
    K = K.view(seq_len,n_heads,int(d_model/n_heads))
    V = V.view(seq_len,n_heads,int(d_model/n_heads))

    Q,K,V = Q.transpose(0,1), K.transpose(0,1), V.transpose(0,1)
    output=self_attention(Q,K,V)
    output = output.transpose(0, 1)   # (seq_len, n_heads, d_head)
    output = output.contiguous().view(seq_len, d_model)
    return output
    
    
    