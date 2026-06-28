"""
Lecture 1: Coding Neurons and Layers
====================================
Building neural network components from scratch using pure Python
lists and loops — no NumPy, just fundamental operations.

Topics covered:
  - Single neuron: weighted sum + bias
  - Layer of neurons using list-of-lists
  - Layer with a batch of inputs
  - Comparing with NumPy for verification
"""


def single_neuron(inputs, weights, bias):
    """Compute the output of a single neuron.

    A neuron computes: output = sum(inputs[i] * weights[i]) + bias

    Args:
        inputs: List of input values
        weights: List of weight values (same length as inputs)
        bias: Bias term (scalar)

    Returns:
        float: The neuron's output
    """
    output = 0
    for i in range(len(inputs)):
        output += inputs[i] * weights[i]
    output += bias
    return output


def layer_forward(inputs, weights, biases):
    """Compute the forward pass through a layer of neurons.

    Each neuron in the layer receives the same inputs but has its
    own weights and bias.

    Args:
        inputs: List of input values (n_inputs,)
        weights: List of lists — one inner list per neuron (n_neurons, n_inputs)
        biases: List of bias values, one per neuron (n_neurons,)

    Returns:
        list: Output values for each neuron in the layer
    """
    layer_output = []
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        layer_output.append(neuron_output)
    return layer_output


def layer_forward_batch(inputs_batch, weights, biases):
    """Compute forward pass for a batch of samples through a layer.

    Args:
        inputs_batch: List of samples, each sample is a list of input values
        weights: List of lists — one inner list per neuron
        biases: List of bias values, one per neuron

    Returns:
        list: Output for each sample, each containing one value per neuron
    """
    batch_output = []
    for sample_inputs in inputs_batch:
        sample_output = layer_forward(sample_inputs, weights, biases)
        batch_output.append(sample_output)
    return batch_output


def main():
    """Demonstrate neurons and layers from scratch."""
    print("=" * 60)
    print("BUILDING NEURAL NETWORKS FROM SCRATCH — Lecture 1")
    print("Neurons and Layers with Pure Python")
    print("=" * 60)

    # ── Example 1: Single Neuron ─────────────────────────────────
    print("\n" + "-" * 40)
    print("[1] SINGLE NEURON")
    print("-" * 40)

    inputs = [1, 2, 3, 2.5]
    weights = [0.2, 0.8, -0.5, 1.0]
    bias = 2.0

    output = single_neuron(inputs, weights, bias)
    print(f"   Inputs: {inputs}")
    print(f"   Weights: {weights}")
    print(f"   Bias: {bias}")
    print(f"   Output: {output:.4f}")

    # ── Example 2: Layer of Neurons ──────────────────────────────
    print("\n" + "-" * 40)
    print("[2] LAYER OF NEURONS")
    print("-" * 40)

    weights_layer = [
        [0.2, 0.8, -0.5, 1.0],      # Neuron 1
        [0.5, -0.91, 0.26, -0.5],    # Neuron 2
        [-0.26, -0.27, 0.17, 0.87]   # Neuron 3
    ]
    biases_layer = [2.0, 3.0, 0.5]

    layer_output = layer_forward(inputs, weights_layer, biases_layer)
    print(f"   Inputs: {inputs}")
    print(f"   Weights (3 neurons × 4 inputs):")
    for i, w in enumerate(weights_layer):
        print(f"      Neuron {i+1}: {w}")
    print(f"   Biases: {biases_layer}")
    print(f"   Layer output: {[f'{v:.4f}' for v in layer_output]}")

    # ── Example 3: Layer with Batch of Inputs ────────────────────
    print("\n" + "-" * 40)
    print("[3] LAYER WITH BATCH OF INPUTS")
    print("-" * 40)

    inputs_batch = [
        [1, 2, 3, 2.5],     # Sample 1
        [2.0, 5.0, -1.0, 2.0],   # Sample 2
        [-1.5, 2.7, 3.3, -0.8]   # Sample 3
    ]

    batch_output = layer_forward_batch(inputs_batch, weights_layer, biases_layer)
    print(f"   Batch of {len(inputs_batch)} samples, each with {len(inputs_batch[0])} features")
    for i, sample_out in enumerate(batch_output):
        print(f"   Sample {i+1}: {[f'{v:.4f}' for v in sample_out]}")

    # ── Example 4: Manual Nested Loops (explicit) ────────────────
    print("\n" + "-" * 40)
    print("[4] EXPLICIT NESTED LOOPS (full control)")
    print("-" * 40)

    X = inputs_batch
    W = weights_layer
    b = biases_layer

    batch_output_explicit = []
    for i in range(len(X)):           # For each sample in the batch
        sample_out = []
        for j in range(len(W)):       # For each neuron in the layer
            neuron_val = 0
            for k in range(len(X[i])):  # For each input feature
                neuron_val += X[i][k] * W[j][k]
            neuron_val += b[j]
            sample_out.append(neuron_val)
        batch_output_explicit.append(sample_out)

    print("   Same result with explicit 3-level nested loops:")
    for i, sample_out in enumerate(batch_output_explicit):
        print(f"   Sample {i+1}: {[f'{v:.4f}' for v in sample_out]}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[OK] KEY INSIGHTS")
    print("=" * 60)
    print("  • A neuron = weighted sum of inputs + bias")
    print("  • A layer = collection of neurons (same inputs, different weights)")
    print("  • Batch = processing multiple samples through the same layer")
    print("  • 3 nested loops = batch loop, neuron loop, feature loop")
    print("  • NumPy will replace these loops with fast matrix operations")
    print("=" * 60)


if __name__ == "__main__":
    main()
