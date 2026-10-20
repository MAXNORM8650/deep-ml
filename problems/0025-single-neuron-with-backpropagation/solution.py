import torch
import torch.nn as nn



def train_neuron(features, labels, initial_weights, initial_bias,
                 learning_rate, epochs):
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
    # Your code here
    features = torch.as_tensor(features, dtype = torch.float32)
    labels = torch.tensor(labels,dtype=torch.float32).reshape(-1)
    weights = torch.as_tensor(initial_weights, dtype=torch.float32,).clone().detach().requires_grad_(True)
    bias = torch.as_tensor(initial_bias,dtype=torch.float32).clone().detach().requires_grad_(True)
    optimizer = torch.optim.SGD([weights,bias], lr=learning_rate)
    mse_values = []
    # loss_fun = torch.nn.MseLoss()

    for epoch in range(epochs):
        loss_all = 0
        for i, feat in enumerate((features)):
            output = torch.sigmoid(torch.matmul(features[i], weights)+bias)
            target = labels[i]
            loss = (output-target) ** 2          
            loss_all+=loss
        loss_all = loss_all / len(features)
        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()
        # mse_values.append(round(loss_all, 4))
        mse_values.append(round(loss_all.item() , 4))
    return [round(w.item(), 4) for w in weights], round(bias.item(), 4), mse_values

