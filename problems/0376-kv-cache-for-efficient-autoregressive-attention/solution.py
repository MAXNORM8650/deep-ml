import torch
import math

def kv_cache_attention_step(x_new: torch.Tensor, W_Q: torch.Tensor, W_K: torch.Tensor, W_V: torch.Tensor, cache: tuple) -> tuple:
    """
    Perform a single attention step with KV caching.
    
    Args:
        x_new: New token embedding, shape (d_model,)
        W_Q: Query projection matrix, shape (d_model, d_k)
        W_K: Key projection matrix, shape (d_model, d_k)
        W_V: Value projection matrix, shape (d_model, d_v)
        cache: Tuple (K_cache, V_cache) of tensors or None if first step
    
    Returns:
        Tuple (output, updated_cache) where output is shape (d_v,)
        and updated_cache is (K_new, V_new)
    """
    q = x_new@W_Q
    k = (x_new@W_K).unsqueeze(0)
    v = (x_new@W_V).unsqueeze(0)
    scale = math.sqrt(q.shape[-1])
    if cache is None:
        K_cache = k
        V_cache = v
    else:
        K_cache,V_cache = cache
        K_cache = torch.cat([K_cache,k], dim=0)
        V_cache = torch.cat([V_cache,v], dim=0)
    
    score = q@K_cache.T/scale

    max_score = torch.max(score, dim=-1, keepdim=True).values
    softmax_ = torch.exp(score-max_score)
    softmax_score = softmax_/torch.sum(softmax_, dim=-1, keepdim=True)
    return softmax_score@V_cache, (K_cache,V_cache)




