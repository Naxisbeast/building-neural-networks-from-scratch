# 🧠 Building Neural Networks from Scratch

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A hands-on journey through neural network fundamentals — building everything from the ground up with Python and NumPy.

## 📚 About This Repository

This repository documents my learning journey through the YouTube lecture series **"Building Neural Networks from Scratch"** by **Dr. Raj Dander (MIT PhD)**.

The core philosophy: **True understanding comes from building**. Instead of treating neural networks as a black box that we interact with through high-level libraries (Keras, PyTorch, TensorFlow), we implement every component ourselves — neuron by neuron, layer by layer.

### 🎯 Why Build From Scratch?

- Demystify the "black box" of deep learning
- Develop intuition for how data flows through a network
- Understand shape transformations, matrix operations, and gradient mechanics
- Build a foundation that makes high-level frameworks more powerful tools
- Gain confidence to debug, extend, and innovate neural architectures

## ✅ Topics Covered (Lectures 1–5)

| Lecture | Topic | Key Concepts |
|:-------:|-------|--------------|
| 1 | Neurons & Layers | Single neuron, layer with list-of-lists, nested loops |
| 2 | NumPy Dot Product | Vector-vector, vector-matrix, matrix-matrix, transpose |
| 3 | Multiple Layers & Forward Pass | Stacking layers, dimension tracking, N-layer forward pass |
| 4 | Dense Layer Class | OOP in Python, `Layer_Dense` class, spiral data generation |
| 5 | Summation & Broadcasting | `np.sum(axis, keepdims)`, broadcasting rules, pitfalls |

### 🔮 Coming Up

- Activation Functions (ReLU, Sigmoid, Tanh)
- Loss Functions (Categorical Cross-Entropy, MSE)
- Backpropagation & Gradient Descent
- Training Loops & Batch Processing
- Regularization (Dropout, L1/L2)
- Deeper architectures & practical projects

## 📁 Repository Structure

```
building-neural-networks-from-scratch/
├── README.md               # ← You are here
├── requirements.txt         # Python dependencies
├── LICENSE                  # MIT License
├── .gitignore               # Python gitignore
├── notes/                   # 📝 Lecture notes
│   ├── README.md
│   ├── 01_neurons_and_layers.md
│   ├── 02_numpy_dot_product.md
│   ├── 03_multiple_layers_forward_pass.md
│   ├── 04_dense_layer_class.md
│   └── 05_array_summation_and_broadcasting.md
├── code/                    # 💻 Runnable code examples
│   ├── chapters/            # Code organized by lecture
│   │   ├── 01_neurons_and_layers.py
│   │   ├── 02_numpy_dot_product.py
│   │   ├── 03_multiple_layers.py
│   │   ├── 04_dense_layer_class.py
│   │   └── 05_broadcasting_demo.py
│   └── utils/               # Shared utilities
│       ├── data_generator.py
│       └── visualization.py
├── projects/                # 🚀 Mini-projects
│   ├── README.md
│   └── spiral_classifier/   # First project
│       ├── spiral_classifier.py
│       └── results/
├── experiments/             # 🧪 Experiment logs & plots
│   └── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Basic familiarity with Python syntax
- Enthusiasm for understanding how neural networks really work!

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/building-neural-networks-from-scratch.git
cd building-neural-networks-from-scratch

# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Code

```bash
# Run a specific chapter's code
python code/chapters/01_neurons_and_layers.py

# Run all examples with verbose output
python code/chapters/04_dense_layer_class.py
```

## 🔑 Key Concepts Learned So Far

1. **Neurons as Weighted Sums** — Each neuron computes `y = Σ(wᵢ · xᵢ) + b`, a linear combination of inputs
2. **Layers as Matrix Operations** — A layer of neurons is just a matrix multiplication: `output = input · W.T + bias`
3. **Forward Pass** — Data flows `input → layer_1 → layer_2 → ... → output`, each step transforming the shape
4. **Dense Layer Abstraction** — Encapsulating weights, biases, and forward logic in a reusable `Layer_Dense` class
5. **Broadcasting** — NumPy's automatic shape alignment that makes bias addition clean and efficient

## 📖 How to Use This Repository

- **Follow the lectures** in order, then read the corresponding notes
- **Run the code** to see the concepts in action
- **Experiment** — modify parameters, change layer sizes, break things on purpose
- **Take the practice questions** seriously — they reveal gaps in understanding

## 🤝 Contributing

This is a personal learning repository, but suggestions and corrections are welcome! Feel free to open an issue or submit a pull request.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dr. Raj Dander (MIT PhD)** — for the excellent lecture series that makes neural networks accessible and hands-on
- The open-source Python ecosystem — NumPy, Matplotlib, and the countless tools that make building from scratch practical

---

<p align="center">
  <strong>🧠 Build to understand. Understand to innovate.</strong>
</p>
