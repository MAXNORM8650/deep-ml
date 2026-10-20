
import torch

def train_neuron(features, labels, initial_weights, initial_bias,
                 learning_rate, epochs):

    features = torch.as_tensor(features, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32).reshape(-1)
    weights = torch.tensor(initial_weights, dtype=torch.float32).clone().detach().requires_grad_(True)
    bias = torch.tensor(initial_bias, dtype=torch.float32).clone().detach().requires_grad_(True)

    optimizer = torch.optim.SGD([weights, bias], lr=learning_rate)
    mse_values = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        output = torch.sigmoid(features @ weights + bias)
        loss = torch.mean((output - labels) ** 2)

        loss.backward()
        optimizer.step()

        mse_values.append(round(loss.item(), 4))

    return [round(w.item(), 4) for w in weights], round(bias.item(), 4), mse_values
