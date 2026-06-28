"""
Lecture 5: Array Summation and Broadcasting Rules
==================================================
Understanding NumPy array summation, the axis parameter,
keepdims, broadcasting rules, and their applications in
neural networks.

Topics covered:
  - np.sum() with axis=0 and axis=1
  - The keepdims parameter (and the pitfall)
  - Broadcasting rules
  - Broadcasting in neural network bias addition
  - Common broadcasting pitfalls
"""

import numpy as np


def main():
    """Demonstrate array summation and broadcasting."""
    print("=" * 60)
    print("BUILDING NEURAL NETWORKS FROM SCRATCH — Lecture 5")
    print("Array Summation and Broadcasting")
    print("=" * 60)

    # ── 1. np.sum() Fundamentals ────────────────────────────────
    print("\n" + "-" * 40)
    print("1  np.sum() FUNDAMENTALS")
    print("-" * 40)

    arr = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])

    print(f"   Array:\n{arr}")
    print(f"   Shape: {arr.shape}")
    print()

    # Sum of all elements
    total = np.sum(arr)
    print(f"   Total sum (all elements): {total}")

    # Sum along axis 0 — collapse rows (sum over samples)
    sum_axis0 = np.sum(arr, axis=0)
    print(f"   Sum axis=0 (column-wise): {sum_axis0}")
    print(f"   Shape after: {sum_axis0.shape}")
    print(f"   Meaning: sum each column down the rows")

    # Sum along axis 1 — collapse columns (sum over features)
    sum_axis1 = np.sum(arr, axis=1)
    print(f"\n   Sum axis=1 (row-wise): {sum_axis1}")
    print(f"   Shape after: {sum_axis1.shape}")
    print(f"   Meaning: sum each row across the columns")

    # ── 2. The keepdims Parameter ────────────────────────────────
    print("\n" + "-" * 40)
    print("2  THE keepdims PARAMETER")
    print("-" * 40)

    print(f"   Original array shape: {arr.shape}")
    print()

    # Without keepdims — dimension collapses
    no_keep = np.sum(arr, axis=1, keepdims=False)
    print(f"   axis=1, keepdims=False:")
    print(f"      Result: {no_keep}")
    print(f"      Shape:  {no_keep.shape}  ← lost a dimension!")

    # With keepdims — dimension preserved as size 1
    with_keep = np.sum(arr, axis=1, keepdims=True)
    print(f"\n   axis=1, keepdims=True:")
    print(f"      Result:\n{with_keep}")
    print(f"      Shape:  {with_keep.shape}  ← dimension preserved!")
    print()
    print(f"   [KEY] keepdims=True preserves the original number of dimensions.")
    print(f"      This is critical for correct broadcasting later.")

    # ── 3. Broadcasting Rules ────────────────────────────────────
    print("\n" + "-" * 40)
    print("3  BROADCASTING RULES")
    print("-" * 40)

    print("   NumPy broadcasting rule:")
    print("   'Two dimensions are compatible when they are equal, ")
    print("    or one of them is 1.'")
    print()
    print("   Shapes are compared from the RIGHTMOST dimension.")

    # Example: Adding a scalar to an array
    print("\n   a) Scalar + Array:")
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = 10
    result = a + b
    print(f"      {a.shape} + {np.array(b).shape} → {result.shape}")
    print(f"      [[1,2,3],[4,5,6]] + 10 = \n      {result}")

    # Example: Adding a 1D array to a 2D array
    print("\n   b) 2D Array + 1D Array (row broadcast):")
    a = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = np.array([10, 20, 30])             # (3,)
    result = a + b
    print(f"      {a.shape} + {b.shape} → {result.shape}")
    print(f"      [[1,2,3],[4,5,6]] + [10,20,30] = \n      {result}")
    print(f"      (3,) broadcasts to (2,3) — the scalar-like dimension stretches")

    # Example: Column + Row
    print("\n   c) Column Vector + Row Vector:")
    col = np.array([[1], [2], [3]])         # (3, 1)
    row = np.array([10, 20, 30, 40])        # (4,) — broadcast to (1, 4)
    result = col + row
    print(f"      {col.shape} + {row.shape} → {result.shape}")
    print(f"      Column:\n{col}")
    print(f"      Row: {row}")
    print(f"      Result:\n{result}")
    print(f"      (3,1) stretches to (3,4) and (1,4) stretches to (3,4)")

    # ── 4. Broadcasting in Neural Networks ───────────────────────
    print("\n" + "-" * 40)
    print("4  BROADCASTING IN NEURAL NETWORKS")
    print("-" * 40)

    # This is exactly how bias addition works in Layer_Dense!
    batch_output = np.array([
        [0.5, -0.2, 0.8],     # Sample 1 output (3 neurons)
        [1.2, -0.5, 0.3],     # Sample 2
        [-0.8, 0.9, 0.1],     # Sample 3
        [0.3, 0.7, -0.4]      # Sample 4
    ])  # Shape: (4, 3)

    # Bias vector — shape (1, 3) for proper broadcasting
    bias = np.array([[0.1, -0.1, 0.2]])

    output_with_bias = batch_output + bias
    print(f"   Batch output shape: {batch_output.shape}")
    print(f"   Bias shape:         {bias.shape}")
    print(f"   Result shape:       {output_with_bias.shape}")
    print()
    print(f"   Batch output:\n{batch_output}")
    print(f"\n   Bias:\n{bias}")
    print(f"\n   Result (output + bias):\n{output_with_bias}")
    print()
    print(f"   [KEY] The bias is added to EVERY sample in the batch!")
    print(f"      This is broadcasting in action.")

    # ── 5. The keepdims Pitfall in Neural Networks ───────────────
    print("\n" + "-" * 40)
    print("5  THE keepdims PITFALL")
    print("-" * 40)

    # Simulating a softmax denominator calculation
    layer_outputs = np.array([
        [1.0, 2.0, 0.5],     # Scores for sample 1
        [0.5, 1.5, 3.0],     # Scores for sample 2
        [2.0, 1.0, 0.5]      # Scores for sample 3
    ])  # Shape: (3, 3)

    # The exponential (for softmax)
    exp_values = np.exp(layer_outputs)
    print(f"   exp(outputs):\n{exp_values}")

    # BUG: without keepdims
    sum_bug = np.sum(exp_values, axis=1)
    print(f"\n   Sum without keepdims:")
    print(f"      Result: {sum_bug}")
    print(f"      Shape:  {sum_bug.shape}")
    try:
        softmax_bug = exp_values / sum_bug  # Shape (3,3) / (3,)
        print(f"      Division result: {softmax_bug}")  # May be wrong!
    except ValueError as e:
        print(f"      Error: {e}")

    # FIX: with keepdims
    sum_fix = np.sum(exp_values, axis=1, keepdims=True)
    print(f"\n   Sum WITH keepdims:")
    print(f"      Result:\n{sum_fix}")
    print(f"      Shape:  {sum_fix.shape}")
    softmax_fix = exp_values / sum_fix  # Shape (3,3) / (3,1)
    print(f"      Division result:\n{softmax_fix}")

    print(f"\n   [KEY] Without keepdims: (3,3) / (3,) — misaligned shapes!")
    print(f"      With keepdims:    (3,3) / (3,1) — broadcasts correctly!")

    # ── 6. Common Broadcasting Pitfalls ──────────────────────────
    print("\n" + "-" * 40)
    print("6  COMMON PITFALLS & EDGE CASES")
    print("-" * 40)

    # Pitfall 1: Mismatched shapes
    print("   Pitfall 1 — Shape Mismatch:")
    a = np.ones((3, 4))
    b = np.ones((3,))
    try:
        _ = a + b
    except ValueError as e:
        print(f"      a(3,4) + b(3,) — Error: {e}")
        print(f"      Reason: Compare (3,4) vs (3,) from right: 4 vs 3 → mismatch")

    # Pitfall 2: Forgetting to make bias (1, n_neurons)
    print("\n   Pitfall 2 — Wrong Bias Shape:")
    batch = np.random.randn(5, 10)  # 5 samples, 10 features
    bias_wrong = np.random.randn(10, 1)  # Shape (10, 1) instead of (1, 10)
    bias_right = np.random.randn(1, 10)  # Shape (1, 10)
    print(f"      batch shape: {batch.shape}")
    print(f"      bias (wrong, 10,1): {bias_wrong.shape}")
    print(f"      bias (right, 1,10): {bias_right.shape}")
    result_wrong = batch + bias_wrong  # (5,10) + (10,1) = (5,10) — actually works!
    result_right = batch + bias_right
    print(f"      Result with (10,1): {result_wrong.shape}")
    print(f"      Result with (1,10): {result_right.shape}")
    print(f"      ⚠ (10,1) also broadcasts! But it's adding column-wise,")
    print(f"         not row-wise as intended for bias in neural networks.")

    # ── 7. Practical Visualization ──────────────────────────────
    print("\n" + "-" * 40)
    print("7  VISUALIZING BROADCASTING — Step by Step")
    print("-" * 40)

    print("   How (3,1) + (4,) becomes (3,4):")
    print()
    col = np.array([[1], [2], [3]])
    row = np.array([10, 20, 30, 40])
    print(f"   Step 1: Start with shapes (3,1) and (4,)")
    print(f"   Step 2: Align from right: compare 1 vs 4 → compatible (stretch 1→4)")
    print(f"   Step 3: Next dimension: 3 vs — (missing) → compatible (treat as 1→3)")
    print(f"   Step 4: Both arrays stretch to shape (3,4)")
    print()
    print(f"   Column (3,1) stretches to (3,4):")
    col_stretched = np.broadcast_to(col, (3, 4))
    print(f"   {col_stretched}")
    print(f"\n   Row (1,4) stretches to (3,4):")
    row_stretched = np.broadcast_to(row.reshape(1, 4), (3, 4))
    print(f"   {row_stretched}")
    print(f"\n   Result = Column + Row = ")
    print(f"   {col + row}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[OK] KEY INSIGHTS")
    print("=" * 60)
    print("  • np.sum(axis=0) = sum over rows (samples), np.sum(axis=1) = sum over columns (features)")
    print("  • keepdims=True preserves dimensions — critical for later broadcasting")
    print("  • Broadcasting aligns shapes from the RIGHTMOST dimension")
    print("  • Two dimensions are compatible if equal or one is 1")
    print("  • Neural network bias: shape (1, n_neurons) for correct row-wise broadcast")
    print("  • The keepdims pitfall is a common source of subtle bugs")
    print("=" * 60)


if __name__ == "__main__":
    main()
