"""
[CYCLONE] Spiral Classifier — Project 1
=================================
A neural network that classifies 2D spiral data using dense layers
built entirely from scratch.

This project brings together concepts from Lectures 1-5:
  - Layer_Dense class (Lecture 4)
  - Forward pass through multiple layers (Lecture 3)
  - NumPy dot product operations (Lecture 2)
  - Broadcasting for bias addition (Lecture 5)
  - Spiral data generation (Lecture 4)

Current status: Forward pass only (no training yet).
Future updates will add activation functions, loss computation,
backpropagation, and training.

Author: Learning from Dr. Raj Dander's "Building Neural Networks from Scratch"
"""

import numpy as np
import sys
import os

# Add the code directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))
from utils.data_generator import generate_spiral_data
from utils.visualization import plot_spiral_data


class Layer_Dense:
    """A fully-connected (dense) layer for a neural network.

    Each neuron in the layer receives input from every neuron
    in the previous layer. This is the building block of our network.

    Attributes:
        weights: Weight matrix, shape (n_inputs, n_neurons)
        biases: Bias vector, shape (1, n_neurons)
        output: Cached output from forward pass
    """

    def __init__(self, n_inputs, n_neurons):
        """Initialize the layer with small random weights and zero biases.

        Args:
            n_inputs: Number of input features
            n_neurons: Number of neurons in this layer
        """
        # Small random weights (0.10 * standard normal)
        # Shape: (n_inputs, n_neurons) so forward pass = dot(inputs, weights)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)

        # Zero biases — they'll learn during training
        # Shape: (1, n_neurons) for correct broadcasting
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Compute the forward pass: output = inputs · weights + bias.

        Args:
            inputs: Input data, shape (batch_size, n_inputs)

        Stores:
            self.output: Layer output, shape (batch_size, n_neurons)
        """
        self.output = np.dot(inputs, self.weights) + self.biases


def main():
    """Run the spiral classifier demonstration."""
    # Set seed for reproducibility
    np.random.seed(42)

    print("=" * 60)
    print("[CYCLONE] SPIRAL CLASSIFIER — Building Neural Networks from Scratch")
    print("=" * 60)

    # ── Step 1: Generate the Spiral Dataset ──────────────────────
    print("\n" + "-" * 40)
    print("[CHART] STEP 1: Generate Spiral Data")
    print("-" * 40)

    # Parameters
    N_SAMPLES = 100     # Points per spiral arm
    N_CLASSES = 3       # Number of spiral arms

    X, y = generate_spiral_data(N_SAMPLES, N_CLASSES)

    print(f"   Dataset generated!")
    print(f"   Total samples: {X.shape[0]}")
    print(f"   Features per sample: {X.shape[1]}")
    print(f"   Number of classes: {N_CLASSES}")
    print(f"   Class distribution: {np.bincount(y)}")
    print(f"\n   Sample data (first 5):")
    print(f"   {'Index':>5} {'x':>8} {'y':>8} {'class':>6}")
    print(f"   {'-'*29}")
    for i in range(5):
        print(f"   {i:>5} {X[i,0]:>8.4f} {X[i,1]:>8.4f} {y[i]:>6}")

    # ── Step 2: Build the Network ────────────────────────────────
    print("\n" + "-" * 40)
    print("[BUILD]  STEP 2: Build the Network")
    print("-" * 40)

    # Network architecture: 2 inputs → 5 hidden → 3 outputs
    # 2 inputs because the spiral data has (x, y) coordinates
    # 5 neurons in hidden layer for learning capacity
    # 3 outputs because we have 3 classes

    dense1 = Layer_Dense(n_inputs=2, n_neurons=5)
    dense2 = Layer_Dense(n_inputs=5, n_neurons=3)

    print(f"   Layer 1: 2 inputs → 5 neurons")
    print(f"   Layer 2: 5 inputs → 3 neurons (output)")
    print(f"\n   Layer 1 weights shape: {dense1.weights.shape}")
    print(f"   Layer 1 biases shape:  {dense1.biases.shape}")
    print(f"   Layer 2 weights shape: {dense2.weights.shape}")
    print(f"   Layer 2 biases shape:  {dense2.biases.shape}")

    # ── Step 3: Forward Pass ─────────────────────────────────────
    print("\n" + "-" * 40)
    print("[>]  STEP 3: Forward Pass")
    print("-" * 40)

    # Data flows through the network
    dense1.forward(X)
    print(f"   After Layer 1:")
    print(f"      Input shape:  {X.shape}")
    print(f"      Output shape: {dense1.output.shape}")

    dense2.forward(dense1.output)
    print(f"\n   After Layer 2 (output):")
    print(f"      Input shape:  {dense1.output.shape}")
    print(f"      Output shape: {dense2.output.shape}")

    # ── Step 4: Examine the Output ───────────────────────────────
    print("\n" + "-" * 40)
    print("[SEARCH] STEP 4: Examine Network Output")
    print("-" * 40)

    print(f"   Our network produces 3 scores per sample — one per class.")
    print(f"   Higher score = network thinks that class is more likely.\n")

    print(f"   {'Sample':>6} {'True Class':>12} {'Score C0':>10} {'Score C1':>10} {'Score C2':>10} {'Predicted':>10}")
    print(f"   {'-'*60}")
    for i in range(10):
        true_class = y[i]
        scores = dense2.output[i]
        predicted = np.argmax(scores)
        print(f"   {i:>6} {true_class:>12} {scores[0]:>10.4f} {scores[1]:>10.4f} {scores[2]:>10.4f} {predicted:>10}")

    # ── Step 5: Calculate "Accuracy" (Random Baseline) ──────────
    print("\n" + "-" * 40)
    print("📈 STEP 5: Current Performance")
    print("-" * 40)

    # Without training, the network has random weights
    # Expected accuracy ≈ 33% (random chance for 3 classes)
    predictions = np.argmax(dense2.output, axis=1)
    correct = np.sum(predictions == y)
    total = len(y)
    accuracy = correct / total * 100

    print(f"   Predictions: {predictions[:20]}")
    print(f"   True labels: {y[:20]}")
    print(f"\n   Correct predictions: {correct}/{total}")
    print(f"   Accuracy: {accuracy:.1f}%")
    print(f"   Random chance: {100/N_CLASSES:.1f}%")
    print(f"\n   ⚠ The network has NOT been trained yet!")
    print(f"   With random weights, accuracy is at chance level.")
    print(f"   Training (coming in future lectures) will improve this.")

    # ── Step 6: Save Visualization ───────────────────────────────
    print("\n" + "-" * 40)
    print("💾 STEP 6: Save Dataset Visualization")
    print("-" * 40)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'spiral_dataset.png')

    try:
        plot_spiral_data(X, y, save_path=save_path)
        print(f"   Spiral dataset plot saved to: {save_path}")
    except ImportError:
        print(f"   Matplotlib not available — skipping plot.")
        print(f"   Install with: pip install matplotlib")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[OK] PROJECT SUMMARY")
    print("=" * 60)
    print("  • Spiral dataset generated: 300 points, 3 classes")
    print("  • Network: 2 → 5 → 3 (2 dense layers)")
    print("  • Forward pass completed successfully")
    print("  • Current accuracy: ~33% (random — network untrained)")
    print("  • Activation functions, loss, and training coming next!")
    print("=" * 60)


if __name__ == "__main__":
    main()
