
import torch
import math

def flash_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size: int = 2
) -> torch.Tensor:
    """
    Compute attention using tiled Flash Attention v1 with online softmax.
    """
    seq_len, d_model = Q.shape
    scale = 1.0 / math.sqrt(d_model)

    output = torch.zeros_like(Q)

    for i in range(0, seq_len, block_size):
        q = Q[i:i + block_size]
        bq = q.shape[0]

        # Running maximum, normalization sum, and output accumulator
        m = torch.full(
            (bq, 1), float("-inf"), dtype=Q.dtype, device=Q.device
        )
        l = torch.zeros((bq, 1), dtype=Q.dtype, device=Q.device)
        acc = torch.zeros(
            (bq, V.shape[1]), dtype=V.dtype, device=V.device
        )

        for j in range(0, seq_len, block_size):
            k = K[j:j + block_size]
            v = V[j:j + block_size]

            # Attention scores for this tile
            scores = (q @ k.T) * scale

            # Online softmax update
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)

            p = torch.exp(scores - m_new)
            alpha = torch.exp(m - m_new)

            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + p @ v

            m = m_new

        output[i:i + bq] = acc / l

    return output
