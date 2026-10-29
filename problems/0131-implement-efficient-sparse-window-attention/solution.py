import torch
import math
def sparse_window_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, window_size: int, scale_factor: float = None) -> torch.Tensor:
    """Computes sparse attention with a sliding window mask.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
        window_size: The radius of the attention window
        scale_factor: Scaling factor for the dot product. If None, uses sqrt(d_k).
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    seq_len, d_q = Q.shape
    # Start = max(0, i-m)
    # End = min(seq_len, i+m-1)
    if scale_factor is None:
        scale_factor = math.sqrt(d_q)
    
    outputs = torch.zeros(seq_len, V.shape[-1])

    for i in range(seq_len):
        start = max(0,i-window_size)
        end = min(seq_len, i+window_size+1)

        score = (Q[i,:]@K[start:end,:].T)/scale_factor

        softmax_score = torch.nn.functional.softmax(score, dim=-1)
        outputs[i] = softmax_score@V[start:end,:]
    return outputs

