# Lecture 4: Dense Layer Class (Object-Oriented Programming)

> **Topics:** OOP basics (class, instance, `self`, `__init__`), reusable dense layer, spiral dataset generation
> **Key Insight:** Encapsulating layer logic in a class is the first step toward building a neural network library

---

## 🎯 Key Concepts

### Why Object-Oriented Programming?

Up to now, we've managed weights, biases, and forward pass logic as separate pieces. This becomes **unmanageable** with more layers:

- Each layer needs its own weights and biases
- Forward pass logic is duplicated
- Adding/removing layers requires manual code changes

A **class** bundles the data (weights, bias) and behavior (forward pass) together.

### Python OOP Refresher

| Concept | Description | Example |
|---------|-------------|---------|
| `class` | Blueprint for creating objects | `class Layer_Dense:` |
| `__init__` | Constructor — runs when object is created | `def __init__(self, n_inputs, n_neurons):` |
| `self` | Reference to the specific instance | `self.weights = ...` |
| Method | Function that belongs to a class | `def forward(self, inputs):` |

### The Dense Layer Class

```python
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """Initialize weights and biases for a dense layer."""
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Compute forward pass: output = inputs · weights + bias."""
        self.output = np.dot(inputs, self.weights) + self.biases
```

**Note:** We initialize with `weights` of shape `(n_inputs, n_neurons)` — this avoids needing `.T` later! The forward pass becomes `dot(inputs, weights)` instead of `dot(inputs, weights.T)`.

---

## 💻 Code Walkthrough

### 1. The Layer_Dense Class

```python
import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        """Initialize the layer with random weights and zero biases.

        Args:
            n_inputs: Number of input features (size of input vector)
            n_neurons: Number of neurons in this layer
        """
        # Shape: (n_inputs, n_neurons) — transposed from earlier lectures
        # This means we can do dot(inputs, weights) without .T
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Compute the forward pass through this layer.

        Args:
            inputs: Input data, shape (batch_size, n_inputs)
        """
        self.output = np.dot(inputs, self.weights) + self.biases
```

### 2. Using the Class

```python
# Create the network
layer1 = Layer_Dense(4, 5)   # 4 inputs → 5 neurons (hidden)
layer2 = Layer_Dense(5, 2)   # 5 inputs → 2 neurons (output)

# Sample input
X = np.array([
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
])

# Forward pass — clean and readable!
layer1.forward(X)
layer2.forward(layer1.output)

print(f"Layer 1 output shape: {layer1.output.shape}")  # (3, 5)
print(f"Layer 2 output shape: {layer2.output.shape}")  # (3, 2)
print(f"Final output:\n{layer2.output}")
```

### 3. Understanding the Weight Initialization

```python
# Why 0.10 * np.random.randn(n_inputs, n_neurons)?
np.random.randn(n_inputs, n_neurons)  # Standard normal (mean=0, std=1)
```

- `randn()` generates values from a standard normal distribution
- Multiplying by `0.10` keeps weights **small** (centered around 0, ±0.1)
- Small weights prevent neurons from saturating (when we add activations)
- The distribution is symmetric — positive and negative values equally likely

### 4. The Spiral Dataset

```python
def generate_spiral_data(n_samples, n_classes):
    """Generate a 2D spiral dataset for classification.

    Creates n_classes interlocking spiral arms in 2D space.
    This is a classic non-linearly separable dataset.
    """
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype='uint8')

    for class_number in range(n_classes):
        ix = range(n_samples * class_number, n_samples * (class_number + 1))
        r = np.linspace(0.0, 1, n_samples)
        t = np.linspace(
            class_number * 4,
            (class_number + 1) * 4,
            n_samples
        ) + np.random.randn(n_samples) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number

    return X, y
```

---

## 🔑 Key Takeaways

1. **Encapsulation** — Bundling weights, bias, and forward pass in a class makes the code modular and scalable
2. **Weight shape convention** — Storing weights as `(n_inputs, n_neurons)` avoids needing `.T` during forward pass
3. **Small random initialization** — `0.10 * randn()` keeps initial weights small and balanced
4. **Zero bias initialization** — Biases start at 0 and learn over time
5. **Spiral data** — A 2D dataset that's non-linearly separable, used to demonstrate why neural networks need multiple layers and activation functions

## 📝 Practice Questions

1. Why do we initialize biases as `np.zeros((1, n_neurons))` instead of just `np.zeros(n_neurons)`?
2. What would happen if you initialized all weights to zero?
3. How does the spiral dataset differ from a linearly separable dataset like two Gaussian clusters?
4. Modify the class to store the input shape as well — what attribute would you add?

<details>
<summary>Answers</summary>

1. Shape `(1, n_neurons)` enables **broadcasting** with batched inputs of shape `(batch_size, n_neurons)` — the bias gets added to every sample in the batch
2. All neurons in a layer would compute **identical outputs** and **identical gradients**, making them redundant. This is called the "symmetry problem."
3. Spiral data is **non-linearly separable** — no single straight line can separate the classes. Two Gaussian clusters can be separated by a straight line (linear decision boundary).
4. Add `self.input_shape = (1, n_inputs)` in `__init__` and store it.

</details>

## 📚 Next Steps

Lecture 5 covers **array summation and broadcasting** — critical for understanding how bias addition works, how to sum along specific axes, and the `keepdims` pitfall. These concepts become essential when we implement loss functions and backpropagation.

---

*"We're building our own library. Each class we create is a building block for the neural networks we'll construct."* — Dr. Raj Dander
