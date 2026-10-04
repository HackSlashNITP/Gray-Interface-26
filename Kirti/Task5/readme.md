# MNIST Digit Classification from Scratch (NumPy)

This repository contains my implementation of a 2-layer Artificial Neural Network (ANN) built entirely from scratch using **NumPy** to classify handwritten digits from the MNIST dataset.

## Approach

Instead of using PyTorch or TensorFlow, all core operations—forward propagation, activation functions, loss computation, backpropagation, and parameter updates—were implemented using linear algebra.

- **Input Layer:** 784 nodes ($28 \times 28$ flattened grayscale pixels, normalized to $[0, 1]$).
- **Hidden Layer:** 128 units with **ReLU** activation.
- **Output Layer:** 10 units (digits 0–9) with **Softmax** activation.
- **Loss Function:** Categorical Cross-Entropy Loss.
- **Optimization:** Mini-Batch Gradient Descent (Batch size = 64).

---

## Observations

I conducted 3 experiments to see how hidden layer size, batch size, and learning rate affect convergence and performance[cite: 2]:

| Experiment | Architecture | Learning Rate | Batch Size | Validation Accuracy | Observations |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp 1 (Baseline)** | 784 → 128 → 10 | 0.10 | 64 | ~96.5% | Balanced training speed and high accuracy. |
| **Exp 2** | 784 → 64 → 10 | 0.01 | 32 | ~92.1% | Slower convergence due to small learning rate. |
| **Exp 3** | 784 → 256 → 10 | 0.10 | 64 | ~97.2% | Slightly better capacity, but took longer per epoch. |

---


## Key Learnings

1. **Weight Initialization Matters:** Initializing weights to zero prevents the network from learning distinct features. 
2. **Backpropagation Math:** Implementing the calculus chain rule manually gave me a strong intuitive understanding of how gradients flow through layers.