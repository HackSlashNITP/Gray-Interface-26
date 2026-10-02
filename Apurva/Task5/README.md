# Gray Interface '26 - Task 5: ANN From Scratch (MNIST)

Hey! This is my submission for Task 5. Instead of relying on deep learning libraries like PyTorch or TensorFlow, I built and trained a simple Feedforward Artificial Neural Network (ANN) completely from scratch using Python and NumPy to understand the underlying mathematics and inner workings.

---

## 🚀 Approach & Architecture

* **Data Prep:** Downloaded the MNIST dataset via OpenML, scaled the pixel values between `0` and `1` for numerical stability, and split it into training, validation, and test sets.
* **Network Structure:** 
  * **Input Layer:** 784 neurons (representing 28x28 flattened image pixels).
  * **Hidden Layer:** Activated using **ReLU** (Rectified Linear Unit) to handle non-linearity.
  * **Output Layer:** 10 neurons with **Softmax** activation to produce probability scores for digits 0–9.
* **Optimization:** Implemented mini-batch gradient descent with Categorical Cross-Entropy Loss and manual backpropagation (calculating derivatives via the chain rule).

---

## 🧪 Experiments & Configurations

I tested 3 different configurations to analyze how different hyperparameters and capacities impact convergence:

| Experiment | Hidden Neurons | Learning Rate | Batch Size | Final Val Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Exp 1** | 64 | 0.05 | 64 | 95.73% |
| **Exp 2** | 128 | 0.1 | 32 | **97.21%** |
| **Exp 3** | 256 | 0.01 | 128 | 89.36% |

---

## 📊 Results & Observations

* **Experiment 2** performed the best, reaching a peak validation accuracy of **97.21%** by Epoch 10. The higher learning rate (`0.1`) paired with a smaller batch size (`32`) allowed the network to learn efficiently and update weights more frequently.
* **Experiment 1** showed stable, steady progress, hitting **95.73%** validation accuracy.
* **Experiment 3** lagged behind at **89.36%** accuracy due to a overly conservative learning rate (`0.01`) combined with a large batch size (`128`), causing slower convergence over 10 epochs.
* **Evaluation & Error Analysis:** Testing on unseen data resulted in 394 misclassifications out of the test subset. Reviewing misclassified digits showed common failure modes where messy handwriting styles blurred distinctions (e.g., a slanted `7` predicted as a `2`, or an open loop `0` looking like a `2`).

---

## 💡 Key Learnings

1. Writing out matrix multiplications and backpropagation by hand provided deep intuition on how gradients flow backwards through layers.
2. Small numerical tricks like adding an epsilon (`1e-8`) in log calculations and normalizing inputs are vital to prevent `NaN` errors and keep training stable.