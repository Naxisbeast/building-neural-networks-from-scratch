"""
Lecture 2: NumPy Dot Product
=============================
Replacing pure Python loops with NumPy's vectorized operations
for neural network computations.

Topics covered:
  - Vector-vector dot product (single neuron)
  - Vector-matrix dot product (layer, one sample)
  - Matrix-matrix dot product (layer, batch of samples)
  - The role of transpose in neural networks
  - Performance comparison
"""

import numpy as np
import time


def dot_product_python(a, b):
    """Compute dot product of two vectors using pure Python.

    Args:
        a: First vector (list of numbers)
        b: Second vector (list of numbers, same length as a)

    Returns:
        float: Dot product a · b = sum(a[i] * b[i])
    """
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def main():
    """Demonstrate NumPy dot product operations for neural networks."""
    print("=" * 60)
    print("BUILDING NEURAL NETWORKS FROM SCRATCH — Lecture 2")
    print("NumPy Dot Product for Neural Networks")
    print("=" * 60)

    # ── Example 1: Vector-Vector Dot Product (Single Neuron) ─────
    print("\n" + "-" * 40)
    print("1  VECTOR-VECTOR DOT PRODUCT (Single Neuron)")
    print("-" * 40)

    inputs = np.array([1.0, 2.0, 3.0, 2.5])
    weights = np.array([0.2, 0.8, -0.5, 1.0])
    bias = 2.0

    # Pure Python version
    python_result = dot_product_python(inputs, weights) + bias

    # NumPy version
    numpy_result = np.dot(inputs, weights) + bias

    print(f"   Inputs: {inputs}")
    print(f"   Weights: {weights}")
    print(f"   Bias: {bias}")
    print(f"   Python dot product + bias: {python_result:.4f}")
    print(f"   NumPy dot product + bias:  {numpy_result:.4f}")
    print(f"   Match: {np.isclose(python_result, numpy_result)}")

    # ── Example 2: Layer of Neurons (Vector-Matrix) ──────────────
    print("\n" + "-" * 40)
    print("2  LAYER OF NEURONS (Vector-Matrix Dot Product)")
    print("-" * 40)

    inputs = np.array([1.0, 2.0, 3.0, 2.5])

    weights = np.array([
        [0.2, 0.8, -0.5, 1.0],      # Neuron 1
        [0.5, -0.91, 0.26, -0.5],   # Neuron 2
        [-0.26, -0.27, 0.17, 0.87]  # Neuron 3
    ])

    biases = np.array([2.0, 3.0, 0.5])

    print(f"   Input shape: {inputs.shape}")
    print(f"   Weights shape: {weights.shape}")
    print(f"   Weights.T shape: {weights.T.shape}")
    print()

    # Correct approach: transpose weights so inputs align with neuron columns
    layer_output = np.dot(inputs, weights.T) + biases
    print(f"   Layer output: {layer_output}")
    print(f"   Output shape: {layer_output.shape}")
    print(f"\n   Why weights.T?")
    print(f"   weights has shape (3, 4) — 3 neurons, 4 inputs each")
    print(f"   weights.T has shape (4, 3) — aligned for dot with inputs (4,)")

    # ── Example 3: Batch of Inputs (Matrix-Matrix) ───────────────
    print("\n" + "-" * 40)
    print("3  BATCH OF INPUTS (Matrix-Matrix Dot Product)")
    print("-" * 40)

    inputs_batch = np.array([
        [1.0, 2.0, 3.0, 2.5],      # Sample 1
        [2.0, 5.0, -1.0, 2.0],     # Sample 2
        [-1.5, 2.7, 3.3, -0.8]     # Sample 3
    ])

    print(f"   Input batch shape: {inputs_batch.shape}")
    print(f"   Weights shape: {weights.shape}")

    layer_output = np.dot(inputs_batch, weights.T) + biases
    print(f"\n   Layer output:\n{layer_output}")
    print(f"   Output shape: {layer_output.shape}")
    print(f"   (3 samples × 3 neurons)")

    # ── Example 4: Why Transpose Matters ─────────────────────────
    print("\n" + "-" * 40)
    print("4  WHY TRANSPOSE? — A Visual Explanation")
    print("-" * 40)

    print("   Without transpose: np.dot(inputs, weights)")
    print(f"      ({inputs_batch.shape[0]}, {inputs_batch.shape[1]}) · ({weights.shape[0]}, {weights.shape[1]})")
    print(f"      Inner dims: {inputs_batch.shape[1]} vs {weights.shape[0]} → {inputs_batch.shape[1] == weights.shape[0]}")
    print(f"      Result: ({inputs_batch.shape[0]}, {weights.shape[1]})")
    result_no_transpose = np.dot(inputs_batch, weights)
    print(f"      Output: {result_no_transpose}")
    print(f"      Shape: {result_no_transpose.shape}")
    print()

    print("   With transpose: np.dot(inputs, weights.T)")
    print(f"      ({inputs_batch.shape[0]}, {inputs_batch.shape[1]}) · ({weights.T.shape[0]}, {weights.T.shape[1]})")
    print(f"      Inner dims: {inputs_batch.shape[1]} vs {weights.T.shape[0]} → YES")
    print(f"      Result: ({inputs_batch.shape[0]}, {weights.T.shape[1]})")
    result_transpose = np.dot(inputs_batch, weights.T)
    print(f"      Output: {result_transpose}")
    print(f"      Shape: {result_transpose.shape}")
    print()
    print("   ⚠ The difference is REAL, not just shape cosmetic.")
    print("   Without transpose: each row of W multiplies the INPUTS.")
    print("   With transpose:    each COLUMN of W (neuron) multiplies the INPUTS.")

    # ── Example 5: Speed Comparison ──────────────────────────────
    print("\n" + "-" * 40)
    print("5  SPEED COMPARISON (NumPy vs Pure Python)")
    print("-" * 40)

    # Larger data for meaningful timing
    large_inputs = np.random.randn(100, 64).tolist()  # 100 samples, 64 features
    large_weights = np.random.randn(32, 64).tolist()   # 32 neurons
    large_biases = np.random.randn(32).tolist()

    large_inputs_np = np.array(large_inputs)
    large_weights_np = np.array(large_weights)
    large_biases_np = np.array(large_biases)

    num_iterations = 100

    # Time pure Python
    start = time.time()
    for _ in range(num_iterations):
        _ = layer_forward_batch(large_inputs, large_weights, large_biases)
    python_time = time.time() - start

    # Time NumPy
    start = time.time()
    for _ in range(num_iterations):
        _ = np.dot(large_inputs_np, large_weights_np.T) + large_biases_np
    numpy_time = time.time() - start

    print(f"   Array sizes: batch=100, features=64, neurons=32")
    print(f"   Iterations: {num_iterations}")
    print(f"   Pure Python: {python_time:.4f}s")
    print(f"   NumPy:       {numpy_time:.4f}s")
    print(f"   Speedup:     {python_time / numpy_time:.1f}x")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[OK] KEY INSIGHTS")
    print("=" * 60)
    print("  • np.dot() replaces nested loops with C-optimized operations")
    print("  • Transpose (W.T) aligns weight columns with input features")
    print("  • Matrix-matrix dot = entire batch through a layer at once")
    print("  • NumPy is 10-100x faster than pure Python at scale")
    print("  • Shape compatibility is checked automatically")
    print("=" * 60)


def layer_forward_batch(inputs_batch, weights, biases):
    """Pure Python batch forward pass (for speed comparison)."""
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
    return batch_output


if __name__ == "__main__":
    main()
