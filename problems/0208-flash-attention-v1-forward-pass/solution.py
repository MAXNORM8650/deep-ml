import torch
import math
def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                           block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    # Your code here
    assert len(Q.shape)==2, "only two dim atm"
    seq_len, D_q = Q.shape
    scale = math.sqrt(D_q)
    output = torch.zeros_like(Q)
    for i in range(0,seq_len,block_size):
        q = Q[i:i+block_size]
        seq_len_sm,d_q = q.shape

        r_max = torch.full((seq_len_sm,1), float("-inf"), dtype=Q.dtype,device=Q.device)
        r_sum = torch.zeros((seq_len_sm,1), dtype=Q.dtype,device=Q.device)
        r_accum =  torch.zeros((seq_len_sm, d_q), dtype=Q.dtype,device=Q.device)

        for j in range(0,seq_len,block_size):
            k = K[j:j+block_size]
            v = V[j:j+block_size]
            score = (q@k.T)/scale
            r_max_new = torch.maximum(r_max, torch.max(score, dim=-1, keepdim=True).values)
            central_exp = torch.exp(score-r_max_new)
            delta = torch.exp(r_max-r_max_new)
            r_sum = r_sum*delta+torch.sum(central_exp,dim=-1, keepdim=True)
            r_accum = r_accum*delta+central_exp@v
            r_max = r_max_new
        output[i:i+block_size] = r_accum/r_sum
    return output
        



