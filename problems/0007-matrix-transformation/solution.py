import torch

def transform_matrix(A, T, S) -> torch.Tensor:
    """
    Perform the change-of-basis transform T⁻¹ A S and round to 3 decimals using PyTorch.
    Inputs A, T, S can be Python lists, NumPy arrays, or torch Tensors.
    Returns a 2×2 tensor or tensor(-1.) if T or S is singular.
    """
    A_t = torch.as_tensor(A, dtype=torch.float)
    T_t = torch.as_tensor(T, dtype=torch.float)
    S_t = torch.as_tensor(S, dtype=torch.float)
    # Your implementation here
    det_T = T_t[0][0]*T_t[1][1]-T_t[0][1]*T_t[1][0]
    det_S = S_t[0][0]*S_t[1][1]-S_t[0][1]*S_t[1][0]
    if det_S==0 or det_T==0:
        return torch.tensor(-1.0)
    adj_T = torch.eye(2)
    adj_T[0][0] = T_t[1][1]
    adj_T[1][1] = T_t[0][0]
    adj_T[0][1] = T_t[1][0]
    adj_T[1][0] = T_t[0][1]
    T_inv = 1/det_T*adj_T
    return torch.round(T_inv@(A_t@S_t),decimals=3)
