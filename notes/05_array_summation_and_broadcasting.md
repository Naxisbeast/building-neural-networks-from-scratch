# Lecture 5: Array Summation and Broadcasting Rules

> **Topics:** `np.sum()` with axis and keepdims, broadcasting rules, common pitfalls, neural network applications
> **Key Insight:** Broadcasting is what makes bias addition "just work" — understanding it prevents subtle bugs

---

## 🎯 Key Concepts

### The Problem: Summing Along Wrong Axes

When you have a 2D array (batch of samples), you often need to sum **along a specific axis**:
- **Axis 0** (rows → down): Sum over the batch — aggregate across samples
- **Axis 1** (columns → across): Sum over features — aggregate across features per sample

```python
arr = np.array([
    [1, 2, 3],   # Sample 1
    [4, 5, 6]    # Sample 2
])

np.sum(arr, axis=0)  # [5, 7, 9]   — column-wise sum (over samples)
np.sum(arr, axis=1)  # [6, 15]     — row-wise sum (over features)
```

### The keepdims Pitfall

```python
sum_axis1 = np.sum(arr, axis=1)        # Shape: (2,)  — 1D array
sum_axis1_keep = np.sum(arr, axis=1, keepdims=True)  # Shape: (2, 1)  — 2D column vector
```

Without `keepdims=True`, summing over axis 1 reduces a `(batch, features)` array to `(batch,)` — you **lose a dimension**. This matters when you want to broadcast the result back against the original array.

### Broadcasting Rules

NumPy broadcasting allows arrays of different shapes to be used in arithmetic operations. The rules:

1. **Align shapes from the right** — Compare dimensions starting from the trailing (rightmost) axis
2. **Dimensions are compatible if** they are equal or one of them is 1
3. **If a dimension is missing**, the array is "stretched" along that dimension

| Array 1 Shape | Array 2 Shape | Result Shape | Compatible? |
|:-------------:|:-------------:|:------------:|:-----------:|
| `(3, 4)` | `(4,)` | `(3, 4)` | ✅ Bias added to each row |
| `(3, 4)` | `(1, 4)` | `(3, 4)` | ✅ Explicit row vector |
| `(3, 4)` | `(3, 1)` | `(3, 4)` | ✅ Explicit column vector |
| `(3, 4)` | `(3,)` | `(3, 4)` | ❌ (3, 4) vs (3,) — compares 4 vs 3? No, compares 4 vs 3 → mismatch! |

Wait — let me correct that last example:
- `(3, 4)` vs `(3,)` — align right: compare 4 vs 3 → **not compatible**!
- NumPy would try to broadcast `(3,)` to match `(3, 4)` but can't

---

## 💻 Code Walkthrough

### 1. Understanding np.sum with axis

```python
import numpy as np

# Create a 2D array
arr = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
])

print(f"Array shape: {arr.shape}")  # (3, 4)

# Sum over all elements
total = np.sum(arr)
print(f"Sum of all elements: {total}")  # 78

# Sum along axis 0 (down the rows — column-wise)
col_sum = np.sum(arr, axis=0)
print(f"Axis 0 sum: {col_sum}")       # [15, 18, 21, 24]
print(f"Shape: {col_sum.shape}")      # (4,)

# Sum along axis 1 (across columns — row-wise)
row_sum = np.sum(arr, axis=1)
print(f"Axis 1 sum: {row_sum}")       # [10, 26, 42]
print(f"Shape: {row_sum.shape}")      # (3,)
```

### 2. The keepdims Parameter

```python
# Without keepdims — dimension is lost
row_sum_no_keep = np.sum(arr, axis=1, keepdims=False)
print(f"No keepdims:  shape {row_sum_no_keep.shape}")  # (3,)

# With keepdims — dimension is preserved as 1
row_sum_keep = np.sum(arr, axis=1, keepdims=True)
print(f"With keepdims: shape {row_sum_keep.shape}")     # (3, 1)

print(f"\nNo keepdims:  {row_sum_no_keep}")
print(f"With keepdims:\n{row_sum_keep}")
```

### 3. Broadcasting Bias Addition

This is why our `Layer_Dense` uses bias shape `(1, n_neurons)`:

```python
batch_output = np.array([
    [0.5, -0.2, 0.8],    # Sample 1
    [1.2, -0.5, 0.3],    # Sample 2
    [-0.8, 0.9, 0.1]     # Sample 3
])  # Shape: (3, 3)

bias = np.array([[0.1, -0.1, 0.2]])  # Shape: (1, 3) — explicitly 2D

# Broadcasting adds bias to EVERY sample
output = batch_output + bias
print(f"Output shape: {output.shape}")  # (3, 3)
print(f"Output:\n{output}")

# Without (1, n_neurons) shape:
bias_1d = np.array([0.1, -0.1, 0.2])   # Shape: (3,)
output_1d = batch_output + bias_1d     # Also works! Broadcasting matches (3,) to (3, 3)
```

### 4. Common Broadcasting Pitfalls

```python
# Pitfall 1: Shape mismatch
a = np.ones((3, 4))
b = np.ones((3,))
try:
    result = a + b  # (3, 4) vs (3,) — compares 4 vs 3 → error!
except ValueError as e:
    print(f"Error: {e}")

# Pitfall 2: Unintended broadcasting
a = np.ones((3, 4))
b = np.ones((4, 3))
result = a + b  # No error! (3, 4) vs (4, 3) — both 1s match? No, 4≠3 and neither is 1
# Actually this DOES raise an error — (3,4) vs (4,3) compares 4≠3 and 3≠4
# But (3,4) vs (1, 4) works, and (3,4) vs (4,1) works

# Pitfall 3: keepdims forgotten
scores = np.array([
    [0.5, 1.2, -0.3],
    [0.1, 0.8, -0.7]
])
softmax_denom = np.sum(np.exp(scores), axis=1)
# softmax_denom shape: (2,) — can't broadcast against (2, 3)!

# Fix with keepdims:
softmax_denom = np.sum(np.exp(scores), axis=1, keepdims=True)
# softmax_denom shape: (2, 1) — broadcasts against (2, 3) perfectly
probabilities = np.exp(scores) / softmax_denom
```

### 5. Visualizing Broadcasting

```python
# Broadcasting: (3, 1) + (4,) = (3, 4)
column = np.array([[1], [2], [3]])     # Shape: (3, 1)
row = np.array([10, 20, 30, 40])       # Shape: (4,)

result = column + row
print(f"Column shape: {column.shape}")
print(f"Row shape: {row.shape}")
print(f"Result shape: {result.shape}")
print(f"Result:\n{result}")

# What happens:
# Column broadcasts to: [[1, 1, 1, 1],    # Stretched along axis 1
#                        [2, 2, 2, 2],
#                        [3, 3, 3, 3]]
# Row broadcasts to:    [[10, 20, 30, 40],  # Stretched along axis 0
#                        [10, 20, 30, 40],
#                        [10, 20, 30, 40]]
```

---

## 🔑 Key Takeaways

1. **`axis=0` sums down the rows** (over the batch), **`axis=1` sums across columns** (over features)
2. **`keepdims=True` preserves the original number of dimensions** — critical for broadcasting
3. **Broadcasting aligns from the right** — the trailing dimensions are compared first
4. **Shapes `(1, N)` and `(N,)` both broadcast** against `(batch, N)` — but `(1, N)` is explicit about intent
5. **The `keepdims` pitfall** is the most common broadcasting bug — losing a dimension when you sum

## 📝 Practice Questions

1. What shape does `np.sum(arr, axis=0, keepdims=True)` return if `arr` has shape `(100, 20)`?
2. Can you broadcast a `(3, 1, 4)` array with a `(1, 5, 1)` array? What shape is the result?
3. Why does `np.zeros(3) + np.zeros(3)` work but `np.ones((3, 4)) + np.ones(3)` fail?
4. In the context of neural networks, when would you sum over axis 0 vs axis 1?

<details>
<summary>Answers</summary>

1. `(1, 20)` — keepdims preserves the dimension; axis 0 becomes size 1
2. Yes! Compare from the right: 4 vs 1 → compatible (stretch to 4), 1 vs 5 → compatible (stretch to 5), 3 vs 1 → compatible (stretch to 3). Result shape: `(3, 5, 4)`
3. `(3, 4)` vs `(3,)` — align right: compare 4 vs 3 → neither is 1 and they're unequal → error. `(3,)` vs `(3,)` works because the shapes match exactly.
4. Sum over **axis 0** (across batch) = total loss across all samples. Sum over **axis 1** (across features) = per-sample total (e.g., sum of probabilities for normalization).

</details>

## 📚 Next Steps

With these array operations mastered, we're ready for:
- **Activation functions** (ReLU, Softmax) — non-linear transformations
- **Loss functions** — measuring how wrong our predictions are
- **Backpropagation** — the chain rule in action, computing gradients
- **Optimization** — gradient descent, updating weights

Understanding `sum`, `axis`, and `broadcasting` is the foundation for all of these.

---

*"When you see keepdims=True, remember: I'm preserving this dimension so I can broadcast later."* — Dr. Raj Dander
