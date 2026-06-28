# 🧪 Experiments

> A playground for exploration, debugging, and deeper understanding.

## Purpose

This directory is where we'll:
- **Run experiments** with different architectures, hyperparameters, and datasets
- **Log results** from training runs for comparison
- **Save visualizations** of network behavior (activation distributions, gradients, etc.)
- **Store model weights** from trained networks
- **Debug and prototype** new ideas before turning them into proper projects

## Current Status

The experiments directory is **empty** — we'll populate it as we progress through the course and begin training networks.

## Future Experiments

| Experiment | Description | When |
|------------|-------------|------|
| Activation Comparison | Compare ReLU, Sigmoid, Tanh on spiral data | After activation functions lecture |
| Learning Rate Sweep | Train same network with different learning rates | After backpropagation lecture |
| Layer Depth Experiment | Compare 1-layer vs 2-layer vs 3-layer networks | After building training loop |
| Weight Initialization | Compare different initialization strategies | During optimization lecture |
| Regularization Demo | See how dropout/L2 affect overfitting | During regularization lecture |
| Batch Size Comparison | Small vs large batch training | After implementing batch processing |

## Directory Structure

```
experiments/
├── README.md              # This file
├── logs/                  # Training logs (future)
├── plots/                 # Experiment visualizations (future)
├── models/                # Saved model weights (future)
└── notebooks/             # Jupyter notebooks for exploration (future)
```

## How to Use

1. Create a subdirectory for each experiment (e.g., `experiments/lr_sweep/`)
2. Include a short README explaining the experiment setup and findings
3. Save plots, logs, and model weights together with the experiment
4. Update this README with a link to your experiment

---

*"The only way to truly understand a neural network is to break it — then fix it."* — Learning by experimenting
