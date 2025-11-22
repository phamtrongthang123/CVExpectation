#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy",
#     "matplotlib",
# ]
# ///

"""
Simple optimization algorithm implementations from scratch.
Demonstrates: Gradient Descent, Momentum, RMSprop, and Adam optimizers.
"""

import numpy as np
import matplotlib.pyplot as plt

print("Optimization Algorithm Implementations")
print("=" * 70)

# ============================================================================
# Test function: Rosenbrock function (classic optimization benchmark)
# ============================================================================
def rosenbrock(x, y):
    """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# ============================================================================
# Optimizer Implementations
# ============================================================================

class GradientDescent:
    """Basic Gradient Descent"""

    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads

class Momentum:
    """Gradient Descent with Momentum"""

    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = np.zeros_like(params)

        self.velocity = self.momentum * self.velocity - self.lr * grads
        return params + self.velocity

class RMSprop:
    """RMSprop optimizer"""

    def __init__(self, lr=0.01, decay=0.9, epsilon=1e-8):
        self.lr = lr
        self.decay = decay
        self.epsilon = epsilon
        self.cache = None

    def step(self, params, grads):
        if self.cache is None:
            self.cache = np.zeros_like(params)

        self.cache = self.decay * self.cache + (1 - self.decay) * grads**2
        return params - self.lr * grads / (np.sqrt(self.cache) + self.epsilon)

class Adam:
    """Adam optimizer"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2

        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update parameters
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

# ============================================================================
# Optimize using different algorithms
# ============================================================================
print("\n[1] Comparing Optimization Algorithms")
print("   Objective: Minimize Rosenbrock function")
print("   Starting point: [-1.0, -1.0]")
print("   Optimum: [1.0, 1.0]")

# Initial parameters
initial_params = np.array([-1.0, -1.0])

# Define optimizers
optimizers = {
    'GD': GradientDescent(lr=0.0001),
    'Momentum': Momentum(lr=0.0001, momentum=0.9),
    'RMSprop': RMSprop(lr=0.001, decay=0.9),
    'Adam': Adam(lr=0.001)
}

# Run optimization
iterations = 1000
results = {}

for name, optimizer in optimizers.items():
    print(f"\n   Running {name}...")

    params = initial_params.copy()
    history = [params.copy()]

    for i in range(iterations):
        # Compute gradient
        grads = rosenbrock_gradient(params[0], params[1])

        # Update parameters
        params = optimizer.step(params, grads)
        history.append(params.copy())

        if (i + 1) % 200 == 0:
            loss = rosenbrock(params[0], params[1])
            print(f"      Iter {i+1:4d}: params=[{params[0]:7.4f}, {params[1]:7.4f}], loss={loss:.6f}")

    final_loss = rosenbrock(params[0], params[1])
    results[name] = {
        'history': np.array(history),
        'final_params': params,
        'final_loss': final_loss
    }

# ============================================================================
# Compare final results
# ============================================================================
print("\n[2] Final Results Comparison")
print("   " + "-" * 66)
print(f"   {'Optimizer':<15} {'Final X':>12} {'Final Y':>12} {'Final Loss':>15}")
print("   " + "-" * 66)

for name, result in results.items():
    x, y = result['final_params']
    loss = result['final_loss']
    print(f"   {name:<15} {x:>12.6f} {y:>12.6f} {loss:>15.6f}")

print("   " + "-" * 66)
print(f"   {'Target':<15} {1.0:>12.6f} {1.0:>12.6f} {0.0:>15.6f}")
print("   " + "-" * 66)

# Find best optimizer
best = min(results.items(), key=lambda x: x[1]['final_loss'])
print(f"\n   🏆 Best performer: {best[0]} (Loss: {best[1]['final_loss']:.6f})")

# ============================================================================
# Visualization
# ============================================================================
print("\n[3] Creating Visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Optimization Algorithm Comparison', fontsize=16, fontweight='bold')

# Create contour plot of Rosenbrock function
x_range = np.linspace(-1.5, 1.5, 200)
y_range = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# Plot 1: Optimization paths
axes[0].contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='gray', alpha=0.4)
axes[0].plot(1, 1, 'r*', markersize=20, label='Optimum', zorder=10)

colors = ['blue', 'green', 'orange', 'red']
for (name, result), color in zip(results.items(), colors):
    history = result['history']
    axes[0].plot(history[:, 0], history[:, 1], '-', color=color,
                label=name, linewidth=2, alpha=0.7)
    axes[0].plot(history[0, 0], history[0, 1], 'o', color=color, markersize=8)

axes[0].set_title('Optimization Paths')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Convergence curves
for (name, result), color in zip(results.items(), colors):
    history = result['history']
    losses = [rosenbrock(p[0], p[1]) for p in history]
    axes[1].plot(losses, color=color, label=name, linewidth=2)

axes[1].set_title('Convergence Curves')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('Loss (log scale)')
axes[1].set_yscale('log')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved visualization: optimization_comparison.png")

# ============================================================================
# Algorithm characteristics
# ============================================================================
print("\n[4] Algorithm Characteristics")
print("   " + "-" * 66)
print("   Gradient Descent:")
print("      - Simplest method, follows steepest descent")
print("      - Can be slow for ill-conditioned problems")
print("      - Sensitive to learning rate")
print()
print("   Momentum:")
print("      - Accelerates in relevant directions")
print("      - Dampens oscillations")
print("      - Better for ravine-like surfaces")
print()
print("   RMSprop:")
print("      - Adapts learning rate per parameter")
print("      - Good for non-stationary objectives")
print("      - Works well with recurrent networks")
print()
print("   Adam:")
print("      - Combines momentum and RMSprop")
print("      - Adaptive per-parameter learning rates")
print("      - Generally robust and efficient")
print("   " + "-" * 66)

print("\n" + "=" * 70)
print("Optimization algorithm comparison completed!")
