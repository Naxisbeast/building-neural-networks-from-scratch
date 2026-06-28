"""
Visualization Utilities
========================
Helper functions for visualizing datasets, neural network
outputs, and experiment results.

Functions:
    plot_spiral_data: Visualize the spiral dataset
    plot_layer_outputs: Compare input and output distributions
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_spiral_data(X, y, title="Spiral Dataset", save_path=None):
    """Plot the spiral dataset colored by class.

    Creates a scatter plot where each point is colored according
    to its class label. This helps visualize the non-linear
    decision boundaries required to separate the spiral arms.

    Args:
        X: Feature matrix, shape (n_samples, 2). Column 0 is x,
           column 1 is y.
        y: Class labels, shape (n_samples,). Values 0, 1, ..., n_classes-1.
        title: Plot title (default: "Spiral Dataset")
        save_path: If provided, saves the figure to this path
                   (e.g., "spiral_plot.png")

    Returns:
        matplotlib.figure.Figure: The figure object for further customization
    """
    n_classes = len(np.unique(y))

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a scatter plot for each class
    colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
    for class_id in range(n_classes):
        mask = y == class_id
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=[colors[class_id]],
            label=f"Class {class_id}",
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5,
            s=30
        )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Feature 1 (x)", fontsize=12)
    ax.set_ylabel("Feature 2 (y)", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_layer_outputs(
    layer1_output,
    layer2_output,
    title="Layer Outputs Comparison",
    save_path=None
):
    """Compare the output distributions of two layers.

    Creates side-by-side histograms showing how the distribution
    of values changes as data flows through the network.

    Args:
        layer1_output: Output from the first layer (numpy array)
        layer2_output: Output from the second layer (numpy array)
        title: Plot title
        save_path: Optional path to save the figure

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Flatten outputs for histogram
    l1_flat = layer1_output.flatten()
    l2_flat = layer2_output.flatten()

    axes[0].hist(l1_flat, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_title(f"Layer 1 Output\nShape: {layer1_output.shape}", fontsize=12)
    axes[0].set_xlabel("Neuron output value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    axes[1].hist(l2_flat, bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_title(f"Layer 2 Output\nShape: {layer2_output.shape}", fontsize=12)
    axes[1].set_xlabel("Neuron output value")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    return fig


def plot_comparison(
    X_spiral,
    y_spiral,
    output_before,
    output_after,
    save_path=None
):
    """Compare network outputs before and after a transformation.

    Useful for visualizing the effect of activation functions
    or other transformations on the spiral data.

    Args:
        X_spiral: Original spiral data
        y_spiral: Class labels
        output_before: Network output before transformation
        output_after: Network output after transformation
        save_path: Optional path to save the figure

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original spiral data
    colors = plt.cm.viridis(np.linspace(0, 1, len(np.unique(y_spiral))))
    for class_id in np.unique(y_spiral):
        mask = y_spiral == class_id
        axes[0].scatter(
            X_spiral[mask, 0],
            X_spiral[mask, 1],
            c=[colors[class_id]],
            label=f"Class {class_id}",
            alpha=0.7,
            s=10
        )
    axes[0].set_title("Original Spiral Data", fontsize=12)
    axes[0].set_aspect('equal')
    axes[0].grid(alpha=0.3)

    # Output before
    axes[1].hist(output_before.flatten(), bins=30, alpha=0.7, color='steelblue')
    axes[1].set_title("Output Before", fontsize=12)
    axes[1].grid(alpha=0.3)

    # Output after
    axes[2].hist(output_after.flatten(), bins=30, alpha=0.7, color='coral')
    axes[2].set_title("Output After", fontsize=12)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Quick test when run directly
    print("Testing visualization utilities...")
    from data_generator import generate_spiral_data

    X, y = generate_spiral_data(100, 3)
    fig = plot_spiral_data(X, y, save_path="spiral_test.png")
    print("Visualization test complete!")
    plt.close(fig)
