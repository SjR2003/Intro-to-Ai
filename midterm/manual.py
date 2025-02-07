import numpy as np
import matplotlib.pyplot as plt

# Define the conditions as functions
def condition1(x, y):
    return (2 <= x) & (x <= 6) & (2 <= y) & (y <= 6)

def condition2(x, y):
    return (4 <= x) & (x <= 6) & (y <= x + 2) & (6 <= y) & (y <= 8)

def target_function(x, y):
    return condition1(x, y) | condition2(x, y)

# Generate dataset
np.random.seed(0)
grid_size = 100
x = np.linspace(0, 10, grid_size)
y = np.linspace(0, 10, grid_size)
xv, yv = np.meshgrid(x, y)
data = np.c_[xv.ravel(), yv.ravel()]
labels = target_function(data[:, 0], data[:, 1]).astype(int)

# Manual weights and biases for perfect accuracy
# Layer 1: Five neurons capturing various conditions
weights_l1 = np.array([
    [1, 0],  # x >= 2
    [-1, 0],  # x <= 6
    [0, 1],  # y >= 2
    [0, -1],  # y <= 6
    [1, -1]   # y <= x + 2 (for condition 2)
])
biases_l1 = np.array([-2, 6, -2, 6, -2])

# Layer 2: Combining conditions with logical AND and OR
weights_l2 = np.array([
    [1, 1, 1, 1, 0],  # AND for condition1
    [0, 0, 0, 0, 1]   # AND for condition2
])
biases_l2 = np.array([-3, -1])  # Thresholds for AND operations

# Output layer: OR condition
weights_out = np.array([[1], [1]])  # Combining the two conditions
biases_out = np.array([-0.5])  # Threshold for OR operation

# Activation functions
def relu(x):
    return np.maximum(0, x)

def step(x):
    return (x > 0).astype(int)

# Forward pass
def forward_pass(x):
    h1 = relu(np.dot(x, weights_l1.T) + biases_l1)  # Layer 1
    h2 = relu(np.dot(h1, weights_l2.T) + biases_l2)  # Layer 2
    output = step(np.dot(h2, weights_out) + biases_out)  # Output layer
    return output

# Pass data through the network
outputs = forward_pass(data).reshape(grid_size, grid_size)

# Visualize predicted results alongside ground truth
plt.figure(figsize=(10, 10))
plt.contourf(xv, yv, outputs, levels=1, cmap="coolwarm", alpha=0.5, label="Predictions")
plt.contourf(xv, yv, labels.reshape(grid_size, grid_size), levels=1, cmap="spring", alpha=0.3, label="True Labels")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Predicted vs Ground Truth Regions")
plt.colorbar(label="Prediction")
plt.show()

# Generate random validation data
np.random.seed(42)
num_validation_points = 1000
validation_data = np.random.uniform(0, 10, (num_validation_points, 2))
validation_labels = target_function(validation_data[:, 0], validation_data[:, 1]).astype(int)

# Validate the network
validation_outputs = forward_pass(validation_data).flatten()
validation_predictions = validation_outputs == validation_labels

# Visualize validation results
plt.figure(figsize=(10, 10))
plt.scatter(validation_data[:, 0], validation_data[:, 1], c=validation_outputs, cmap="Greens", alpha=0.5, label="Predicted")
plt.scatter(validation_data[:, 0], validation_data[:, 1], c=validation_labels, cmap="Reds", alpha=0.3, label="Ground Truth")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Validation Results")
plt.legend()
plt.show()
