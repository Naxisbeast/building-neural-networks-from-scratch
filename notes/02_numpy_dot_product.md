# Lecture 2: NumPy Dot Product

> **Topics:** Vector-vector, vector-matrix, matrix-matrix dot products; transpose; neural network applications
> **Key Insight:** A layer's forward pass is just matrix multiplication

---

## 🎯 Key Concepts

### Why NumPy?

Pure Python nested loops from Lecture 1 are:
- **Slow** — Python loops are interpreted, not optimized
- **Verbose** — 3+ levels of nesting for a simple operation
- **Error-prone** — Easy to mix up indices

NumPy solves this with **vectorized operations** written in C, giving us 10–100x speedups.

### The Dot Product

The **dot product** is the fundamental operation of a neural network layer:

$$a \cdot b = \sum_{i=1}^{n} a_i \cdot b_i$$

Two requirements:
1. Both vectors must have the **same length**
2. They are **aligned** — element-wise multiply, then sum

### Why Transpose Matters

When we represent a layer's weights as a matrix of shape `(n_neurons, n_inputs)`, we need to **transpose** it before multiplying with the input:

$$ \text{output} = \text{inputs} \cdot W^T + \text{bias} $$

This is because:
- Input shape: `(n_inputs,)` (column vector conceptually)
- Weight shape: `(n_neurons, n_inputs)` (each row = one neuron's weights)

After transpose: `W.T` has shape `(n_inputs, n_neurons)`, so the dot product gives shape `(n_neurons,)` — one output per neuron.

---

## 💻 Code Walkthrough

### 1. Vector-Vector Dot Product

```python
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

# Pure Python version
python_output = 0
for i in range(len(inputs)):
    python_output += inputs[i] * weights[i]
python_output += bias

# NumPy version (much cleaner!)
numpy_output = np.dot(inputs, weights) + bias

print(f"Python: {python_output:.4f}")  # 4.8
print(f"NumPy:  {numpy_output:.4f}")   # 4.8
```

### 2. Layer of Neurons (Vector-Matrix)

```python
inputs = [1.0, 2.0, 3.0, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0],   # Neuron 1
    [0.5, -0.91, 0.26, -0.5],  # Neuron 2
    [-0.26, -0.27, 0.17, 0.87]  # Neuron 3
]

biases = [2.0, 3.0, 0.5]

# Without transpose — would fail or give wrong shape
# layer_output = np.dot(weights, inputs) + biases  # Wrong!

# With transpose — correct
layer_output = np.dot(inputs, np.array(weights).T) + biases

print(f"Layer output: {layer_output}")
print(f"Shape: {layer_output.shape}")  # (3,)
```

### 3. Batch Processing (Matrix-Matrix)

```python
# batch of 3 samples, each with 4 features
inputs_batch = np.array([
    [1.0, 2.0, 3.0, 2.5],    # Sample 1
    [2.0, 5.0, -1.0, 2.0],   # Sample 2
    [-1.5, 2.7, 3.3, -0.8]   # Sample 3
])

weights = np.array([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
])

biases = np.array([2.0, 3.0, 0.5])

# Matrix-matrix dot product
layer_output = np.dot(inputs_batch, weights.T) + biases

print(f"Layer output:\n{layer_output}")
print(f"Shape: {layer_output.shape}")  # (3, 3)
```

### 4. Comparing Speeds

```python
import time

# Pure Python
start = time.time()
for _ in range(10000):
    # Nested loops version
    pass
python_time = time.time() - start

# NumPy
start = time.time()
for _ in range(10000):
    np.dot(inputs_batch, weights.T) + biases
numpy_time = time.time() - start

print(f"NumPy is {python_time / numpy_time:.1f}x faster")
```

---

## 📐 Understanding Shapes

The most critical skill for building neural networks:

| Operation | Input Shape | Weight Shape | Output Shape |
|-----------|-------------|--------------|--------------|
| Single neuron | `(n_inputs,)` | `(n_inputs,)` | `(1,)` → scalar |
| Layer (1 sample) | `(n_inputs,)` | `(n_neurons, n_inputs)` | `(n_neurons,)` |
| Layer (batch) | `(batch_size, n_inputs)` | `(n_neurons, n_inputs)` | `(batch_size, n_neurons)` |

**Rule of thumb:** The last dimension of the first matrix must match the second-to-last dimension of the second matrix.

---

## 🔑 Key Takeaways

1. **`np.dot(a, b)`** computes the dot product with automatic shape checking
2. **Transpose** is essential — we flip the weight matrix so inputs align with neuron weights
3. **Vector-matrix dot** computes a whole layer at once (one sample)
4. **Matrix-matrix dot** processes an entire batch of samples simultaneously
5. **NumPy is 10–100x faster** than pure Python loops for these operations

## 📝 Practice Questions

1. What shape is `weights.T` if `weights` has shape `(5, 4)`?
2. Can you take the dot product of a `(3, 4)` matrix and a `(5, 3)` matrix? What shape does the result have?
3. Why do we do `np.dot(inputs, weights.T)` instead of `np.dot(weights, inputs)`?
4. What happens if the batch size changes from 3 to 100? Does the code need to change?

<details>
<summary>Answers</summary>

1. `(4, 5)` — Transpose swaps rows and columns
2. Yes! The inner dimensions match: `(3, 4) · (4, 5)` → `(3, 5)` — wait, `(5, 3)` doesn't match with `(3, 4)`. Actually `(3, 4) · (5, 3)` — the inner dimensions are 4 and 5, which don't match. **No**, you'd need `(3, 4) · (4, 5)`.
3. Because `weights` has shape `(n_neurons, n_inputs)` and we want to multiply each input by each neuron's weights. After transpose, shape is `(n_inputs, n_neurons)`, so `dot(inputs, weights.T)` gives `(n_neurons,)`.
4. No change needed! The matrix-matrix dot product handles any batch size automatically. Output shape would be `(100, 3)`.

</details>

## 📚 Next Steps

Lecture 3 stacks multiple layers together, creating the **forward pass** of a neural network. The shape-tracking skills from this lecture become essential.

---

*"NumPy dot product is doing the exact same thing as our nested loops — just much, much faster."* — Dr. Raj Dander
