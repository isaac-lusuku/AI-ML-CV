import torch
import numpy

# accessing the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# creating the set of inputs and outputs
x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], device=device)  # the inputs
y = torch.tensor([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], device=device)  # the outputs

# initialize the weights
w = torch.tensor(0.0, requires_grad=True, device=device)
learning_rate = torch.tensor(0.01, device=device)


# computing the predictions
def predict(x_):
    return x_ * w


# computing the loss MSE
def loss(x1):
    return ((x1 - y) ** 2).mean()


# training the model
for epoch in range(100):
    predictions = predict(x)
    loss_ = loss(predictions)
    loss_.backward()
    with torch.no_grad():
        w -= (learning_rate * w.grad)

    w.grad.zero_()

print(f"prediction for 33 is: {33 * w}")

