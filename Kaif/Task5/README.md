# MNIST Digit Classification — ANN

## Google Colab Link--

https://colab.research.google.com/drive/1DX2cXK20Lke7uNjJaJqOCVoJnc3XToPa?usp=sharing

## 1. Approach

Built a neural network **using only NumPy**. The pipeline covers EDA, preprocessing, a hand-written forward/backward pass, mini-batch gradient descent training, multiple experiments, and final evaluation with metrics + visualizations.

## 2. Data & Preprocessing

- 42,000 labeled images, each 28×28 grayscale (784 pixel features), split into ~41,000 train / 1,000 validation (dev) samples.
- **Normalization:** pixel values (0–255) are scaled to `[0, 1]` by dividing by 255. Without this, the raw large pixel values push the network's internal values too high, activations get stuck, and it can't learn.
- EDA: checked the class distribution (roughly balanced across digits) and visualized one sample image per class to confirm the data looks correct before training.

## 3. Architecture

A simple **2-layer feedforward network**:

```
Input (784 pixels) → Hidden layer (ReLU) → Output layer (10 classes, Softmax)
```

- **Input layer:** 784 units — one per pixel.
- **Hidden layer:** configurable size (10 or 64 units tested), uses **ReLU** activation `f(z) = max(0, z)` — this is what lets the network learn non-linear patterns instead of just a straight-line decision boundary.
- **Output layer:** 10 units (one per digit), uses **Softmax**, which turns the 10 raw scores into probabilities that sum to 1, so the highest one is the model's predicted digit.
- **Weight initialization:** He initialization (`weights × √(2/fan_in)`) — a standard way to start ReLU networks with weights that are neither too large (exploding) nor too small (vanishing gradients).

## 4. Experiments

Three configurations were compared, varying hidden layer size, learning rate, and batch size:

| Experiment | Hidden units | Learning rate | Batch size | Epochs | Test Accuracy |
|---|---|---|---|---|---|
| Exp 1 | 10 | 0.1 | 64 | 40 | 0.95 |
| Exp 2 | 64 | 0.1 | 64 | 40 | 0.982 |
| Exp 3 | 64 | 0.5 | 128 | 40 | 0.979 |

## 5.Obseravtions

-Hidden Layer Size and Accuracy: tested two hidden layer sizes: 10 and 64 units. Increasing the hidden layer from 10 to 64 units improved validation accuracy from 0.95 to 0.982.
-Learning Rate and Performance: Increasing the learning rate from 0.1 to 0.5 slightly decreased accuracy (from 0.982 to 0.979) and also made the training more unstable, as seen in the loss curves.
-Batch Size and Convergence: Using a larger batch size of 128 (compared to 64) resulted in slightly faster convergence in terms of epochs needed to reach peak performance.
