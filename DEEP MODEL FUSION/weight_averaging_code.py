import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# --- 1. CREATE A SIMPLE FEEDFORWARD NETWORK ---
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 2)  # Binary classification

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # No softmax (handled in loss function)


# --- 2. GENERATE TOY DATASET ---
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


# --- 3. TRAIN TWO MODELS SEPARATELY ---
def train_model(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()


# Create and train two models independently
model1 = SimpleNN(input_size=10)
model2 = SimpleNN(input_size=10)

train_model(model1, train_loader)
train_model(model2, train_loader)


# --- 4. APPLY WEIGHT AVERAGING ---
def average_weights(model1, model2):
    """Averages the weights of two models"""
    avg_model = SimpleNN(input_size=10)
    model1_dict = model1.state_dict()
    model2_dict = model2.state_dict()

    avg_dict = {key: (model1_dict[key] + model2_dict[key]) / 2 for key in model1_dict}
    avg_model.load_state_dict(avg_dict)

    return avg_model


# Fuse the two models
fused_model = average_weights(model1, model2)


# --- 5. EVALUATE PERFORMANCE ---
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predictions = torch.argmax(output, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


# Compare accuracy
acc1 = evaluate_model(model1, test_loader)
acc2 = evaluate_model(model2, test_loader)
acc_fused = evaluate_model(fused_model, test_loader)

print(f"Model 1 Accuracy: {acc1:.4f}")
print(f"Model 2 Accuracy: {acc2:.4f}")
print(f"Fused Model Accuracy: {acc_fused:.4f}")
