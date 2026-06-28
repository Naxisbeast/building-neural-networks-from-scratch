# Lecture 1: Coding Neurons and Layers

> **Topics:** Single neuron computation, layer of neurons using list-of-lists, nested loops
> **Philosophy:** Build from scratch — no NumPy, no high-level libraries, just pure Python lists

---

## 🎯 Key Concepts

### The Biological Analogy

A **neuron** in a neural network is loosely inspired by biological neurons:
- **Inputs** (dendrites) → receive signals from other neurons
- **Processing** (cell body) → computes a weighted sum
- **Output** (axon) → passes the result to the next layer

### The Single Neuron

A single neuron computes a **weighted sum** of its inputs plus a **bias**:

$$y = \sum_{i=1}^{n} (w_i \cdot x_i) + b$$

Where:
- $x_i$ = input values
- $w_i$ = weights (how important each input is)
- $b$ = bias (allows the neuron to shift its activation)
- $y$ = output

### Layer of Neurons

A **layer** is a collection of neurons, all receiving the same inputs. Each neuron has its own set of weights and bias.

```
Inputs        Neuron 1        Neuron 2        Neuron 3
    x₁ ──────── w₁₁ ──────┐     w₁₂ ──────┐     w₁₃ ──────┐
    x₂ ──────── w₂₁ ──────┤     w₂₂ ──────┤     w₂₃ ──────┤
    x₃ ──────── w₃₁ ──────┤     w₃₂ ──────┤     w₃₃ ──────┤
                          ├────► y₁        ├────► y₂        ├────► y₃
    bias ────── b₁ ───────┘   b₂ ─────────┘   b₃ ─────────┘
```

### List-of-Lists Representation

We represent a layer's weights as a **list of lists** (a nested list):
- Outer list: one inner list per **neuron**
- Inner list: one weight per **input**

```python
weights = [
    [0.2, 0.8, -0.5],  # Neuron 1: weights for inputs [x1, x2, x3]
    [0.5, -0.91, 0.26],  # Neuron 2: weights for inputs [x1, x2, x3]
    [-0.26, -0.27, 0.17]  # Neuron 3: weights for inputs [x1, x2, x3]
]
```

---

## 💻 Code Walkthrough

### 1. Single Neuron

```python
inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

output = 0
for i in range(len(inputs)):
    output += inputs[i] * weights[i]
output += bias

print(output)  # 4.8
```

### 2. Layer of Neurons (Nested Loops)

```python
inputs = [1, 2, 3, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0],   # Neuron 1
    [0.5, -0.91, 0.26, -0.5],  # Neuron 2
    [-0.26, -0.27, 0.17, 0.87]  # Neuron 3
]

biases = [2.0, 3.0, 0.5]

# Output of the layer = one value per neuron
layer_output = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)  # [4.8, 1.21, 2.385]
```

### 3. Layer with a Batch of Inputs

```python
# Multiple samples processed through the same layer
inputs_batch = [
    [1, 2, 3, 2.5],  # Sample 1
    [2.0, 5.0, -1.0, 2.0],  # Sample 2
    [-1.5, 2.7, 3.3, -0.8]  # Sample 3
]

batch_output = []
for sample_inputs in inputs_batch:
    sample_output = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(sample_inputs, neuron_weights):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        sample_output.append(neuron_output)
    batch_output.append(sample_output)

print(batch_output)
# [[4.8, 1.21, 2.385],
#  [8.9, -1.81, 0.2],
#  [1.41, 1.051, 0.026]]
```

---

## 🔑 Key Takeaways

1. **A neuron is just a weighted sum + bias** — There's no magic, just multiplication and addition
2. **A layer is multiple neurons** sharing the same inputs, each with independent weights and bias
3. **Nested loops** give us complete control but are slow — this is why we eventually use NumPy
4. **Shape pattern**: Inputs (`n_features`) → Weights (`n_neurons × n_features`) → Outputs (`n_neurons`)
5. **Batch processing** — Processing multiple samples together gives us a 2D output shape

## 📝 Practice Questions

1. If you have 4 inputs and a layer with 5 neurons, how many total weights does the layer have?
2. What changes if you add another input feature to every sample?
3. Write the loop that processes a single neuron without using `zip()`.
4. How would you implement a neuron that uses `inputs[i] * weights[i]` but skips the bias?

<details>
<summary>Answers</summary>

1. 4 inputs × 5 neurons = **20 weights** (plus 5 biases)
2. Every neuron's weight list grows by 1 element; the loop iterates 5 times instead of 4
3. Use index-based loop: `for i in range(len(inputs)): neuron_output += inputs[i] * weights[i]`
4. Simply omit the bias addition: `bias = 0` or skip the `+= bias` line

</details>

## 📚 Next Steps

Lecture 2 replaces these nested loops with NumPy's `dot()` function — much faster and more readable. The concepts stay the same; only the implementation changes.

---

*"We're not using NumPy yet — we want to see what the computer is actually doing."* — Dr. Raj Dander
