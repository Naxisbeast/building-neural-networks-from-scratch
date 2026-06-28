# Lecture 3: Multiple Layers and Forward Pass

> **Topics:** Stacking layers, forward pass formula, dimension tracking, scalable N-layer implementation
> **Key Insight:** A neural network is just a chain of matrix multiplications

---

## 🎯 Key Concepts

### The Architecture of a Neural Network

A neural network with multiple layers creates a **hierarchy of representations**:

```
Input ──► Layer 1 ──► Layer 2 ──► ... ──► Output
(4)        (5)         (3)                 (3)
```

Each layer:
- Receives the **output** of the previous layer as its **input**
- Transforms it through another matrix multiplication
- Passes the result forward

### Forward Pass Formula

For a network with $L$ layers:

$$a^{[0]} = X \text{ (input)}$$
$$z^{[l]} = a^{[l-1]} \cdot W^{[l]T} + b^{[l]}$$
$$a^{[l]} = f(z^{[l]}) \text{ (activation — coming in future lectures)}$$

Where:
- $a^{[l]}$ is the output of layer $l$
- $W^{[l]}$ is the weight matrix of layer $l$
- $b^{[l]}$ is the bias vector of layer $l$

### Shape Tracking

The most important skill: **every layer's output shape must match the next layer's input shape**.

```
Layer 1: (batch, 4) ──dot── W1.T(4, 5) ──► (batch, 5)
Layer 2: (batch, 5) ──dot── W2.T(5, 3) ──► (batch, 3)
```

The **4** in → **5** connects because:
- Layer 1 output = `(batch, 5)` → this becomes the input to Layer 2
- Layer 2 expects input of size **5** (its weight matrix has 5 columns per row)

---

## 💻 Code Walkthrough

### 1. Two-Layer Network (Explicit)

```python
import numpy as np

# Input — a batch of 3 samples, each with 4 features
X = np.array([
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
])

# Layer 1: 4 inputs → 5 neurons
weights1 = np.array([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
    [0.11, 0.32, -0.64, 0.15],
    [0.42, -0.71, 0.33, 0.52]
])  # Shape: (5, 4)

biases1 = np.array([2.0, 3.0, 0.5, 1.0, -0.5])  # Shape: (5,)

# Layer 2: 5 inputs → 2 neurons (output layer)
weights2 = np.array([
    [0.1, -0.2, 0.3, -0.4, 0.5],
    [0.6, -0.7, 0.8, -0.9, 0.1]
])  # Shape: (2, 5)

biases2 = np.array([0.5, -0.3])  # Shape: (2,)

# Forward pass
layer1_output = np.dot(X, weights1.T) + biases1
print(f"Layer 1 output shape: {layer1_output.shape}")  # (3, 5)

layer2_output = np.dot(layer1_output, weights2.T) + biases2
print(f"Layer 2 output shape: {layer2_output.shape}")  # (3, 2)

print(f"\nFinal output:\n{layer2_output}")
```

### 2. Scalable Forward Pass (N Layers)

The key insight: as long as each layer's weight matrix is initialized with the right shape, the forward pass looks the **same for every layer**:

```python
# Define layers as tuples of (weights, biases)
layers = [
    (weights1, biases1),  # Layer 1: 4→5
    (weights2, biases2)   # Layer 2: 5→2
]

# Forward pass — same code works for ANY number of layers!
current_input = X
for i, (W, b) in enumerate(layers):
    current_input = np.dot(current_input, W.T) + b
    print(f"After layer {i+1}: shape {current_input.shape}")

final_output = current_input
```

### 3. Dimension Tracking Utility

```python
def forward_pass_with_trace(X, layers_weights, layers_biases):
    """Compute forward pass with shape logging."""
    current = X
    print(f"{'Layer':>8} {'Input Shape':>20} {'Output Shape':>20}")
    print("-" * 50)
    print(f"{'Input':>8} {'—':>20} {str(X.shape):>20}")

    for i, (W, b) in enumerate(zip(layers_weights, layers_biases)):
        input_shape = current.shape
        current = np.dot(current, W.T) + b
        print(f"Layer {i+1:>2} {str(input_shape):>20} {str(current.shape):>20}")

    return current

output = forward_pass_with_trace(X, [weights1, weights2], [biases1, biases2])
```

---

## 🔑 Key Takeaways

1. **A neural network is a function composition**: `output = layer2(layer1(input))`
2. **Layer shapes must chain correctly** — output size of layer L → input size of layer L+1
3. **The forward pass code is identical for every layer** — just `dot(input, W.T) + b`
4. **A network with two dense layers (no activations) is still just a linear transformation** — the next step is adding non-linear activation functions
5. **Spiral data** is nonlinear — a linear model (no activations) can't classify it properly

## 📝 Practice Questions

1. If layer 1 has 64 neurons and layer 2 has 32 neurons, what must be the output shape of layer 1 and the input size of layer 2?
2. What happens to the forward pass if you accidentally set `weights1` to shape `(4, 3)` instead of `(5, 4)`?
3. How would you add a third hidden layer with 10 neurons to the example above?
4. Why is the forward pass sometimes called "inference"?

<details>
<summary>Answers</summary>

1. Layer 1 output = `(batch, 64)`, so layer 2 must expect input of size **64** (its weight matrix has 64 columns)
2. NumPy raises a shape mismatch error — `np.dot(X, weights1.T)` would try `(3, 4) · (3, 4)` — the inner dimensions (3, 4) and (3, 4) don't actually work. Wait, `weights1.T` would be `(3, 4)` and `X` is `(3, 4)` — so `np.dot(X, weights1.T)` would give `(3, 3)`. Then adding biases of shape `(4,)` would broadcast to `(3, 3)` — silently wrong!
3. Add `weights3 = np.random.randn(10, 64)` and `biases3 = np.random.randn(10)`, then `np.dot(layer2_output, weights3.T) + biases3`
4. Because during inference (prediction), we only run the **forward** pass — no backward pass or gradient computation needed

</details>

## 📚 Next Steps

Lecture 4 introduces the **Dense Layer class** — encapsulating weights, bias, and forward logic into a reusable Python object. This is our first step toward building a real neural network library.

---

*"Each layer just takes what it gets from the previous layer and applies its own transformation."* — Dr. Raj Dander
