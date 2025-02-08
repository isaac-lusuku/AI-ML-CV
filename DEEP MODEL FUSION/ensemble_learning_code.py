import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader


# --- 1. DEFINE A SIMPLE NEURAL NETWORK ---
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


# --- 3. TRAIN MULTIPLE MODELS ---
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


# Train three separate models
models = [SimpleNN(input_size=10) for _ in range(3)]
for i, model in enumerate(models):
    print(f"Training Model {i + 1}...")
    train_model(model, train_loader)


# --- 4. ENSEMBLE LEARNING USING MAJORITY VOTING ---
def ensemble_predict(models, X):
    """Predict using majority voting from multiple models"""
    model_outputs = [model(X) for model in models]
    predictions = [torch.argmax(output, dim=1) for output in model_outputs]

    # Stack predictions from all models
    stacked_preds = torch.stack(predictions, dim=1)  # Shape: [batch_size, num_models]

    # Compute mode (most frequent prediction per sample)
    ensemble_preds, _ = torch.mode(stacked_preds, dim=1)  # Mode returns values & indices
    return ensemble_preds


# --- 5. EVALUATE INDIVIDUAL MODELS AND ENSEMBLE ---
def evaluate_model(model, test_loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = torch.argmax(model(X_batch), dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


def evaluate_ensemble(models, test_loader):
    correct, total = 0, 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            ensemble_preds = ensemble_predict(models, X_batch)
            correct += (ensemble_preds == y_batch).sum().item()
            total += y_batch.size(0)

    return correct / total


# Print individual accuracies
for i, model in enumerate(models):
    acc = evaluate_model(model, test_loader)
    print(f"Model {i + 1} Accuracy: {acc:.4f}")

# Print ensemble accuracy
ensemble_acc = evaluate_ensemble(models, test_loader)
print(f"Ensemble Accuracy (Majority Voting): {ensemble_acc:.4f}")
