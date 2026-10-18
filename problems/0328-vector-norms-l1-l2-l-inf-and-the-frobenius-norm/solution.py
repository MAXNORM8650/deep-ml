import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.

    'l1', 'l2' and 'linf' are entrywise norms and accept a 1D or 2D array.
    'frobenius' is a matrix norm and must raise a ValueError if arr is not 2D.

    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', 'linf', or 'frobenius')

    Returns:
        The computed norm as a float
    """
    # Your code here
    if len(arr.shape)==1 and norm_type=='frobenius':
        raise ValueError("arr is not 2D")
    if norm_type=='l1':
        return float(sum(abs(arr.flatten())))
    elif norm_type=='l2':
        return float(np.sqrt(sum(np.square(arr.flatten()))))
    elif norm_type=='linf':
        return float(np.max(abs(arr.flatten())))
    elif norm_type=='frobenius':
        return float(np.sqrt(np.sum(np.square(arr.flatten()))))
def round(x, r):
    return np.round(x, r)

