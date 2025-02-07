import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score

# Load dataset
file_path = "output_file_name.ext"  # Update with actual file path
data = pd.read_csv(file_path)

# Display basic info and strip column names
data.columns = data.columns.str.strip()
print(data.info())
print(data.describe())

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Sort correlation with the target variable
target_column = "Chance of Admit"  # Update if needed
correlations = data.corr()[target_column].abs().sort_values(ascending=False)
print("Feature correlations with target (descending):")
print(correlations)

# Preprocessing
X = data.drop(columns=[target_column])
y = data[target_column]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# Train-validation split
val_split = int(0.85 * X_train_tensor.shape[0])
X_train_final = X_train_tensor[:val_split]
y_train_final = y_train_tensor[:val_split]
X_val = X_train_tensor[val_split:]
y_val = y_train_tensor[val_split:]

# Model Definitions
class MLP1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP1, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP2, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Device Configuration
device = torch.device("cpu")  # Force CPU for simplicity
input_dim = X_train_tensor.shape[1]
hidden_dim1, hidden_dim2 = 64, 32
output_dim = 1

# Initialize models
model1 = MLP1(input_dim, hidden_dim1, output_dim).to(device)
model2 = MLP2(input_dim, hidden_dim1, hidden_dim2, output_dim).to(device)

# Function for training and storing metrics
def train_model_with_metrics(model, optimizer, X_train, y_train, X_val, y_val, epochs=100, patience=10):
    train_losses = []
    val_losses = []
    train_r2_scores = []
    val_r2_scores = []
    best_val_loss = float("inf")
    best_model_path = f"{model.__class__.__name__}_best_model.pth"
    criterion = nn.MSELoss()
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(X_train)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            train_r2 = r2_score(y_train.cpu(), train_pred.cpu().detach())
            val_r2 = r2_score(y_val.cpu(), val_pred.cpu())

        train_losses.append(train_loss.item())
        val_losses.append(val_loss)
        train_r2_scores.append(train_r2)
        val_r2_scores.append(val_r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss.item()}, Val Loss = {val_loss}, Train R^2 = {train_r2:.4f}, Val R^2 = {val_r2:.4f}")

    return train_losses, val_losses, train_r2_scores, val_r2_scores, best_model_path

# Train models
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

train_losses1, val_losses1, train_r2_1, val_r2_1, model1_path = train_model_with_metrics(
    model1, optimizer1, X_train_final, y_train_final, X_val, y_val
)
train_losses2, val_losses2, train_r2_2, val_r2_2, model2_path = train_model_with_metrics(
    model2, optimizer2, X_train_final, y_train_final, X_val, y_val
)

# Plot Loss and R^2 Metrics
def plot_metrics(train_losses, val_losses, train_r2, val_r2, model_name):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 6))

    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.title(f"Loss Curve: {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # R^2 Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_r2, label="Train R^2")
    plt.plot(epochs, val_r2, label="Validation R^2")
    plt.title(f"R^2 Curve: {model_name}")
    plt.xlabel("Epochs")
    plt.ylabel("R^2")
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_metrics(train_losses1, val_losses1, train_r2_1, val_r2_1, "MLP1")
plot_metrics(train_losses2, val_losses2, train_r2_2, val_r2_2, "MLP2")

# Compare Test Performance
def evaluate_model(model, model_path, X_test, y_test, criterion):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        test_loss = criterion(predictions, y_test).item()
        test_r2 = r2_score(y_test.cpu(), predictions.cpu())
    print(f"Test Loss: {test_loss}, Test R^2: {test_r2:.4f}")
    return test_loss, test_r2

# Call the function with criterion
print("Evaluating Model 1...")
test_loss1, test_r2_1 = evaluate_model(model1, model1_path, X_test_tensor, y_test_tensor, nn.MSELoss())
print("Evaluating Model 2...")
test_loss2, test_r2_2 = evaluate_model(model2, model2_path, X_test_tensor, y_test_tensor, nn.MSELoss())

if test_r2_1 > test_r2_2:
    print("MLP1 is the best model.")
    torch.save(model1.state_dict(), "best_model.pth")  # Save only weights
else:
    print("MLP2 is the best model.")
    torch.save(model2.state_dict(), "best_model.pth")  # Save only weights
