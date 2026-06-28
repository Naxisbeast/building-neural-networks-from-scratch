"""
Lecture 4: Dense Layer Class (Object-Oriented Programming)
===========================================================
Encapsulating layer logic in a reusable Python class.
Introduces the Layer_Dense class and spiral dataset generation.

Topics covered:
  - The Layer_Dense class
  - Weight initialization strategies
  - Building networks with the class
  - Spiral dataset generation
  - Forward pass with OOP
"""

import numpy as np
import sys
import os

# Add the parent directory to the path so we can import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.data_generator import generate_spiral_data


class Layer_Dense:
    """A fully-connected (dense) layer for a neural network.

    Each neuron in the layer receives input from every neuron in the
    previous layer (hence "fully-connected" or "dense").

    Attributes:
        weights: Weight matrix, shape (n_inputs, n_neurons)
        biases: Bias vector, shape (1, n_neurons)
        output: Cached output from the most recent forward pass
    """

    def __init__(self, n_inputs, n_neurons):
        """Initialize the layer with random weights and zero biases.

        Args:
            n_inputs: Number of input features
            n_neurons: Number of neurons in this layer

        Weight initialization: 0.10 * randn() keeps weights small (~±0.1)
        Bias initialization: zeros — biases will learn during training
        """
        # Shape: (n_inputs, n_neurons)
        # This convention means forward pass uses dot(inputs, weights)
        # instead of dot(inputs, weights.T)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        # Shape: (1, n_neurons) — 2D for proper broadcasting with batches
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Compute the forward pass through this layer.

        output = inputs · weights + biases

        Args:
            inputs: Input data, shape (batch_size, n_inputs)

        Stores:
            self.output: Layer output, shape (batch_size, n_neurons)
        """
        self.output = np.dot(inputs, self.weights) + self.biases


def main():
    """Demonstrate the Layer_Dense class."""
    np.random.seed(42)  # For reproducibility
    print("=" * 60)
    print("BUILDING NEURAL NETWORKS FROM SCRATCH — Lecture 4")
    print("Dense Layer Class (OOP)")
    print("=" * 60)

    # ── 1. Creating a Single Dense Layer ─────────────────────────
    print("\n" + "-" * 40)
    print("1  CREATING A DENSE LAYER")
    print("-" * 40)

    layer = Layer_Dense(n_inputs=4, n_neurons=5)

    print(f"   Layer created: 4 inputs → 5 neurons")
    print(f"   Weights shape: {layer.weights.shape}")
    print(f"   Biases shape:  {layer.biases.shape}")
    print(f"\n   Weights (first 2 neurons):")
    print(f"   {layer.weights[:, :2]}")
    print(f"\n   Biases:")
    print(f"   {layer.biases}")

    # ── 2. Forward Pass with a Single Layer ──────────────────────
    print("\n" + "-" * 40)
    print("2  FORWARD PASS — Single Layer")
    print("-" * 40)

    X = np.array([
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]
    ])

    layer.forward(X)
    print(f"   Input shape:  {X.shape}")
    print(f"   Output shape: {layer.output.shape}")
    print(f"   Output:\n{layer.output}")

    # ── 3. Two-Layer Network ─────────────────────────────────────
    print("\n" + "-" * 40)
    print("3  TWO-LAYER NETWORK")
    print("-" * 40)

    # Create layers
    layer1 = Layer_Dense(4, 5)   # Hidden layer: 4 inputs → 5 neurons
    layer2 = Layer_Dense(5, 2)   # Output layer: 5 inputs → 2 neurons

    # Forward pass
    layer1.forward(X)
    layer2.forward(layer1.output)

    print(f"   Layer 1 output shape: {layer1.output.shape}")
    print(f"   Layer 2 output shape: {layer2.output.shape}")
    print(f"\n   Final network output:\n{layer2.output}")

    # ── 4. Larger Network ────────────────────────────────────────
    print("\n" + "-" * 40)
    print("4  LARGER NETWORK")
    print("-" * 40)

    # 100 samples, 10 features each
    X_large = np.random.randn(100, 10)

    # Three-layer network
    hidden1 = Layer_Dense(10, 64)   # 10 → 64
    hidden2 = Layer_Dense(64, 32)   # 64 → 32
    output_layer = Layer_Dense(32, 3)  # 32 → 3

    # Forward pass
    hidden1.forward(X_large)
    hidden2.forward(hidden1.output)
    output_layer.forward(hidden2.output)

    print(f"   Network: 10 → 64 → 32 → 3")
    print(f"   Samples: {X_large.shape[0]}")
    print(f"   Hidden 1 output: {hidden1.output.shape}")
    print(f"   Hidden 2 output: {hidden2.output.shape}")
    print(f"   Final output:    {output_layer.output.shape}")

    # ── 5. Spiral Data ───────────────────────────────────────────
    print("\n" + "-" * 40)
    print("5  SPIRAL DATASET")
    print("-" * 40)

    # Generate spiral data: 100 samples per class, 3 classes
    X_spiral, y_spiral = generate_spiral_data(100, 3)

    print(f"   Generated spiral dataset:")
    print(f"   Samples: {X_spiral.shape[0]}")
    print(f"   Features per sample: {X_spiral.shape[1]}")
    print(f"   Number of classes: {len(np.unique(y_spiral))}")
    print(f"   Class distribution: {np.bincount(y_spiral)}")
    print(f"\n   First 5 samples:")
    for i in range(5):
        print(f"   Sample {i+1}: x={X_spiral[i][0]:.4f}, y={X_spiral[i][1]:.4f}, class={y_spiral[i]}")

    # ── 6. Forward Pass on Spiral Data ───────────────────────────
    print("\n" + "-" * 40)
    print("6  FORWARD PASS ON SPIRAL DATA")
    print("-" * 40)

    # Build network for spiral data (2 features, 3 classes)
    spiral_layer1 = Layer_Dense(2, 4)   # 2 inputs (x, y) → 4 hidden neurons
    spiral_layer2 = Layer_Dense(4, 3)   # 4 hidden → 3 output classes

    # Forward pass
    spiral_layer1.forward(X_spiral)
    spiral_layer2.forward(spiral_layer1.output)

    print(f"   Network: 2 → 4 → 3")
    print(f"   Input:  {X_spiral.shape}")
    print(f"   Layer 1: {spiral_layer1.output.shape}")
    print(f"   Layer 2: {spiral_layer2.output.shape}")
    print(f"\n   Output (first 5 samples):")
    for i in range(5):
        print(f"   Sample {i+1} (class {y_spiral[i]}): {spiral_layer2.output[i]}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[OK] KEY INSIGHTS")
    print("=" * 60)
    print("  • OOP makes neural network code modular and reusable")
    print("  • Layer_Dense.forward() encapsulates dot product + bias")
    print("  • Networks are built by stacking Layer_Dense objects")
    print("  • Weight shape (n_inputs, n_neurons) avoids .T in forward pass")
    print("  • Spiral data is non-linearly separable — needs depth + activations")
    print("  • Next step: activation functions for non-linearity!")
    print("=" * 60)


if __name__ == "__main__":
    main()
