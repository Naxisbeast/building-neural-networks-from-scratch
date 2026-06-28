# 🌀 Spiral Classifier

> A neural network implementation that classifies points in a 2D spiral dataset using dense layers built from scratch.

## Overview

The spiral dataset is a classic non-linearly separable problem — no single straight line can separate the colored arms. This project implements a forward pass through a multi-layer network to demonstrate how even a simple linear network transforms the input space.

## Problem Statement

Given a 2D point (x, y), predict which of 3 spiral arms it belongs to. The spiral arms are generated using polar coordinates with added noise, creating interlocking arcs that require a non-linear decision boundary.

## Dataset Characteristics

- **Number of samples**: 300 (100 per class)
- **Number of features**: 2 (x and y coordinates)
- **Number of classes**: 3 (spiral arms)
- **Data type**: Non-linearly separable
- **Noise**: Gaussian noise added to angles (±0.2 rad)

## Network Architecture

```
Input (2 features)
    ↓
Dense Layer 1 — 2 inputs → 5 neurons
    ↓ (weights: 2×5, bias: 1×5)
Dense Layer 2 — 5 inputs → 3 neurons (output)
    ↓ (weights: 5×3, bias: 1×3)
Output (3 class scores)
```

## How It Works

1. **Data Generation**: Uses `generate_spiral_data()` to create 300 points in 3 interlocking spiral arms
2. **Layer Creation**: Two `Layer_Dense` instances form the network
3. **Forward Pass**: Data flows through both layers, transforming from 2D space to 3D class scores
4. **Visualization**: Output shows the raw scores for each sample across all 3 classes

## Current Limitations

- **No activation functions** — the network is purely linear (stacked matrix multiplications)
- **No training** — weights are random, no learning occurs yet
- **No loss computation** — we can't measure how "wrong" the predictions are

These limitations will be addressed in future lectures (activations, loss functions, backpropagation).

## Future Enhancements

- [ ] Add ReLU activation after hidden layers
- [ ] Add Softmax activation on output
- [ ] Implement loss computation (Categorical Cross-Entropy)
- [ ] Add backpropagation and gradient descent
- [ ] Train the network and visualize decision boundaries
- [ ] Compare accuracy with and without activations

## How to Run

```bash
# From the repository root
python projects/spiral_classifier/spiral_classifier.py

# Or navigate to the project directory
cd projects/spiral_classifier
python spiral_classifier.py
```

## Expected Output

The script prints:
- Dataset information (shapes, class distribution)
- Network architecture details
- Forward pass output for the first 5 samples
- Output shapes at each layer

---

*"This spiral dataset will be our testbed throughout the course — we'll keep coming back to it as we add new capabilities."* — Dr. Raj Dander
