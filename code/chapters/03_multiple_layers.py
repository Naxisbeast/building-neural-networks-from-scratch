"""
Lecture 3: Multiple Layers and Forward Pass
============================================
Stacking layers to build a simple neural network and tracking
shapes through the forward pass.

Topics covered:
  - Two-layer network forward pass
  - Dimension tracking through the network
  - Scalable N-layer forward pass
  - Visualizing layer transformations
"""

import numpy as np


def forward_pass_with_trace(X, layers_weights, layers_biases):
    """Compute forward pass with detailed shape logging.

    Args:
        X: Input data, shape (batch_size, n_features)
        layers_weights: List of weight matrices
        layers_biases: List of bias vectors

    Returns:
        numpy.ndarray: Final layer output
    """
    current = X

    print(f"{'Stage':>10} {'Input Shape':>20} {'Output Shape':>20}")
    print("-" * 55)
    print(f"{'Input':>10} {'—':>20} {str(X.shape):>20}")

    for i, (W, b) in enumerate(zip(layers_weights, layers_biases)):
        input_shape = current.shape
        current = np.dot(current, W.T) + b
        print(f"Layer {i+1:>2}  {str(input_shape):>20} {str(current.shape):>20}")

    return current


def main():
    """Demonstrate multi-layer forward pass."""
    np.random.seed(42)  # For reproducibility
    print("=" * 60)
    print("BUILDING NEURAL NETWORKS FROM SCRATCH — Lecture 3")
    print("Multiple Layers and Forward Pass")
    print("=" * 60)

    # ── Input Data ───────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("[CHART] INPUT DATA")
    print("-" * 40)

    X = np.array([
        [1.0, 2.0, 3.0, 2.5],          # Sample 1
        [2.0, 5.0, -1.0, 2.0],         # Sample 2
        [-1.5, 2.7, 3.3, -0.8],        # Sample 3
        [0.5, -1.2, 3.1, 1.5],         # Sample 4
        [3.0, 1.0, -2.0, 0.5]          # Sample 5
    ])

    print(f"   Number of samples: {X.shape[0]}")
    print(f"   Number of features per sample: {X.shape[1]}")
    print(f"   Input shape: {X.shape}")
    print(f"\n   Sample 1: {X[0]}")
    print(f"   Sample 2: {X[1]}")
    print(f"   Sample 3: {X[2]}")

    # ── Network Architecture ─────────────────────────────────────
    print("\n" + "-" * 40)
    print("[BUILD]  NETWORK ARCHITECTURE")
    print("-" * 40)

    architecture = [
        ("Input", 4),
        ("Hidden Layer 1", 5),
        ("Output Layer", 3)
    ]

    print(f"   {'Layer':<20} {'Neurons':<10} {'Weights Shape':<15}")
    print(f"   {'-'*45}")
    for name, neurons in architecture:
        print(f"   {name:<20} {neurons:<10}")

    # ── Initialize Weights and Biases ────────────────────────────
    print("\n" + "-" * 40)
    print("🎲 WEIGHT INITIALIZATION")
    print("-" * 40)

    # Layer 1: 4 inputs → 5 neurons
    weights1 = np.array([
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
        [0.11, 0.32, -0.64, 0.15],
        [0.42, -0.71, 0.33, 0.52]
    ])
    biases1 = np.array([2.0, 3.0, 0.5, 1.0, -0.5])

    # Layer 2: 5 inputs → 3 neurons (output)
    weights2 = np.array([
        [0.1, -0.2, 0.3, -0.4, 0.5],
        [0.6, -0.7, 0.8, -0.9, 0.1],
        [-0.3, 0.4, -0.5, 0.6, -0.7]
    ])
    biases2 = np.array([0.5, -0.3, 0.8])

    print(f"   Layer 1 weights shape:  {weights1.shape}  (5 neurons × 4 inputs)")
    print(f"   Layer 1 biases shape:   {biases1.shape}")
    print(f"   Layer 2 weights shape:  {weights2.shape}  (3 neurons × 5 inputs)")
    print(f"   Layer 2 biases shape:   {biases2.shape}")

    # ── Forward Pass ─────────────────────────────────────────────
    print("\n" + "-" * 40)
    print("[>]  FORWARD PASS")
    print("-" * 40)

    # Manual two-layer forward pass
    print("\n   Manual 2-layer forward pass:")
    print(f"   {'Step':<30} {'Shape':<15}")
    print(f"   {'-'*45}")

    layer1_output = np.dot(X, weights1.T) + biases1
    print(f"   {'After Layer 1 (4 → 5):':<30} {str(layer1_output.shape):<15}")

    layer2_output = np.dot(layer1_output, weights2.T) + biases2
    print(f"   {'After Layer 2 (5 → 3):':<30} {str(layer2_output.shape):<15}")

    print(f"\n   Layer 1 output:\n{layer1_output}")
    print(f"\n   Layer 2 output (final):\n{layer2_output}")

    # ── Forward Pass with Trace ──────────────────────────────────
    print("\n" + "-" * 40)
    print("[SEARCH] FORWARD PASS WITH DIMENSION TRACKING")
    print("-" * 40)

    layers_weights = [weights1, weights2]
    layers_biases = [biases1, biases2]

    final_output = forward_pass_with_trace(X, layers_weights, layers_biases)

    print(f"\n   Final output:\n{final_output}")

    # ── Scalable Architecture ────────────────────────────────────
    print("\n" + "-" * 40)
    print("[BUILD]  SCALABLE N-LAYER ARCHITECTURE")
    print("-" * 40)

    # Define any architecture as a list of (n_inputs, n_neurons) tuples
    architecture_config = [
        (4, 8),     # Layer 1: 4 inputs → 8 neurons
        (8, 6),     # Layer 2: 8 inputs → 6 neurons
        (6, 6),     # Layer 3: 6 inputs → 6 neurons
        (6, 3),     # Layer 4: 6 inputs → 3 neurons (output)
    ]

    print(f"   Architecture: 4 → 8 → 6 → 6 → 3")
    print(f"   (input → hidden1 → hidden2 → hidden3 → output)")

    # Build layers dynamically
    layers_weights_dynamic = []
    layers_biases_dynamic = []
    for n_inputs, n_neurons in architecture_config:
        W = 0.10 * np.random.randn(n_inputs, n_neurons)
        b = np.zeros((1, n_neurons))
        layers_weights_dynamic.append(W)
        layers_biases_dynamic.append(b)

    print(f"\n   Dynamically created {len(layers_weights_dynamic)} layers:")
    for i, W in enumerate(layers_weights_dynamic):
        print(f"   Layer {i+1}: weights {W.shape}, biases {layers_biases_dynamic[i].shape}")

    # Forward pass through dynamic network
    print(f"\n   Forward pass through {len(layers_weights_dynamic)}-layer network:")
    current = X
    for i, (W, b) in enumerate(zip(layers_weights_dynamic, layers_biases_dynamic)):
        current = np.dot(current, W) + b
        print(f"   After layer {i+1}: {current.shape}")

    print(f"\n   Final output shape: {current.shape}")
    print(f"   Final output:\n{current}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[OK] KEY INSIGHTS")
    print("=" * 60)
    print("  • A neural network = chain of linear transformations")
    print("  • Each layer transforms the shape: (batch, in) → (batch, out)")
    print("  • The forward pass is identical for every layer")
    print("  • Dimensions must chain: L1_out == L2_in == L3_in ...")
    print("  • Without activations, N layers = 1 big linear transformation")
    print("  • Next step: add non-linear activation functions!")
    print("=" * 60)


if __name__ == "__main__":
    main()
