import torch
import torch.nn as nn

def train_neuron(features: torch.Tensor, labels: torch.Tensor, initial_weights: torch.Tensor, initial_bias: float, learning_rate: float, epochs: int) -> tuple[list[float], float, list[float]]:
    """
    Simulates a single neuron with sigmoid activation and trains it using
    backpropagation with MSE loss via SGD.

    Args:
        features: Input feature tensor of shape (n_samples, n_features)
        labels: Binary label tensor of shape (n_samples,)
        initial_weights: Initial weight tensor of shape (n_features,)
        initial_bias: Initial bias scalar
        learning_rate: Learning rate for SGD
        epochs: Number of training epochs

    Returns:
        Tuple of (updated_weights, updated_bias, mse_values) all rounded to 4 decimal places
    """
    # 1. Initialize parameters as PyTorch leaf tensors tracking gradients
    # Ensure inputs match the requested types
    w = torch.as_tensor(initial_weights, dtype=torch.float32).clone().detach().requires_grad_(True)
    b = torch.tensor(initial_bias, dtype=torch.float32, requires_grad=True)
    
    # Cast dataset tensors safely
    X = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)

    # Define loss function and the SGD optimizer explicitly managing our parameters
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD([w, b], lr=learning_rate)
    
    mse_values = []

    for ep in range(epochs):
        # --- Batch Forward Pass ---
        # 1. Linear combination: z = Xw + b
        # 2. Sigmoid activation: a = 1 / (1 + exp(-z))
        predictions = torch.sigmoid(torch.matmul(X, w) + b)
        
        # Calculate full-batch loss for tracking metrics
        loss = criterion(predictions, y)
        mse_values.append(round(loss.item(), 4))
        
        # --- Backpropagation Step ---
        optimizer.zero_grad()    # Clear gradients from previous epoch
        loss.backward()          # Automatically calculate gradients dw and db
        optimizer.step()         # Apply gradient descent update: param = param - lr * grad

    # 2. Extract final states, convert to Python lists/scalars, and round values
    updated_weights = [round(val, 4) for val in w.detach().tolist()]
    updated_bias = round(b.item(), 4)
    
    return updated_weights, updated_bias, mse_values
