#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy",
# ]
# ///

"""
Simple neural network implementation from scratch using only NumPy.
Demonstrates: forward pass, backpropagation, and gradient descent without frameworks.
"""

import numpy as np

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Simple Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize network with random weights"""
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        """Backward pass and weight update"""
        m = X.shape[0]

        # Output layer gradients
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)

        # Hidden layer gradients
        dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        """Training loop"""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Calculate loss (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))

            # Backward pass
            self.backward(X, y, output, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

        return output

# Generate synthetic data (XOR problem)
np.random.seed(42)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)  # XOR labels

print("Training Neural Network from Scratch")
print("=" * 60)
print(f"Architecture: {X.shape[1]} -> 4 -> {y.shape[1]}")
print(f"Dataset: {X.shape[0]} samples")
print()

# Create and train network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
predictions = nn.train(X, y, epochs=1000, learning_rate=0.5)

print("\nFinal Predictions:")
for i, (input_val, true_val, pred_val) in enumerate(zip(X, y, predictions)):
    print(f"Input: {input_val} | True: {true_val[0]:.0f} | Predicted: {pred_val[0]:.4f} | "
          f"Class: {int(pred_val[0] > 0.5)}")

# Test forward pass manually
print("\nTest inference:")
test_input = np.array([[0, 1]])
test_output = nn.forward(test_input)
print(f"Input: {test_input[0]} -> Output: {test_output[0][0]:.4f} -> Class: {int(test_output[0][0] > 0.5)}")
