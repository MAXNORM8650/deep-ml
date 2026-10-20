import torch

def calculate_eigenvalues(matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvalues of a 2x2 matrix using PyTorch.
    Input: 2x2 tensor; Output: 1-D tensor with the two eigenvalues in descending order (highest to lowest).
    """
    # Your implementation here
    def sol_quadatic_eqn(b,c):
        return torch.tensor((b+torch.sqrt(b**2-4*c))/2), torch.tensor((b-torch.sqrt(b**2-4*c))/2)
    trace_a = matrix[0][0]+matrix[1][1]
    det_a = matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    return torch.tensor(sol_quadatic_eqn(trace_a,det_a))