import numpy as np

def softmax(x):
    exp = np.exp(x)
    return exp/np.sum(exp, axis=1, keepdims=True)
    
def simple_self_attention(X: list[list[float]]) -> list[list[float]]:
    """
    Compute context vectors using simple self-attention (no trainable weights).

    Args:
        X: Input embeddings of shape (T, d) as a list of lists.

    Returns:
        Context vectors of shape (T, d) as a list of lists.
    """
    X = np.array(X)
    d = X.shape[-1]
    score = X@np.transpose(X)#/np.sqrt(d)
    softmax_score = softmax(score)
    return list(softmax_score@X)
