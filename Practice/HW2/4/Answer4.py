import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

class RBFLayer(nn.Module):
    def __init__(self, n_centers, input_dim, gamma):
        super(RBFLayer, self).__init__()
        self.centers = nn.Parameter(torch.empty(n_centers, input_dim).uniform_(0, 1))
        self.gamma = gamma

    def forward(self, x):
        x_expanded = x.unsqueeze(1)  # Expand dimensions for broadcasting
        diff = x_expanded - self.centers
        l2 = torch.sum(diff ** 2, dim=2)
        return torch.exp(-self.gamma * l2)

def load_and_preprocess_data():
    data = fetch_california_housing()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), \
           torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

def build_rbf_model(input_dim, n_centers, gamma):
    class RBFModel(nn.Module):
        def __init__(self):
            super(RBFModel, self).__init__()
            self.rbf = RBFLayer(n_centers, input_dim, gamma)
            self.linear = nn.Linear(n_centers, 1)

        def forward(self, x):
            rbf_output = self.rbf(x)
            return self.linear(rbf_output)

    return RBFModel()

def build_dense_model(input_dim):
    class DenseModel(nn.Module):
        def __init__(self):
            super(DenseModel, self).__init__()
            self.hidden1 = nn.Linear(input_dim, 64)
            self.hidden2 = nn.Linear(64, 32)
            self.output = nn.Linear(32, 1)

        def forward(self, x):
            x = torch.relu(self.hidden1(x))
            x = torch.relu(self.hidden2(x))
            return self.output(x)

    return DenseModel()

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test).item()
            val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

def plot_loss(train_losses, val_losses, title):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    input_dim = X_train.shape[1]

    # RBF Model
    rbf_model = build_rbf_model(input_dim=input_dim, n_centers=50, gamma=0.1)
    print("\nTraining RBF Model")
    train_losses, val_losses = train_and_evaluate_model(rbf_model, X_train, y_train, X_test, y_test)
    plot_loss(train_losses, val_losses, 'RBF Model Loss')

    # Dense Model
    dense_model = build_dense_model(input_dim=input_dim)
    print("\nTraining Dense Model")
    train_losses, val_losses = train_and_evaluate_model(dense_model, X_train, y_train, X_test, y_test)
    plot_loss(train_losses, val_losses, 'Dense Model Loss')

if __name__ == "__main__":
    main()
