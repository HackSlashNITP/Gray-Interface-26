# Task 5: ANN from scratch on MNIST

Notebook: <paste public Colab link here>

A fully connected neural network written in NumPy (no TensorFlow, PyTorch or Keras), trained on the Kaggle Digit Recognizer data. Final test accuracy is 97.08%.

## Approach

I used only `train.csv` (42,000 labelled images). `test.csv` has no labels, so accuracy, F1 and a confusion matrix can't be computed on it. I split `train.csv` into 80/10/10 train/validation/test (33,595 / 4,196 / 4,209), stratified by digit with a fixed seed. Pixels are scaled to [0, 1] and labels are one-hot encoded. Classes are balanced (roughly 9% to 11% each), so plain accuracy is a fair headline metric. The validation split picked the best experiment, and the test split was used once at the end.

## Network

Input 784, then one or more ReLU hidden layers, then a 10-way softmax. Weights use He initialization, biases start at zero. The baseline is 784 → 128 → 10.

Forward pass, per layer: `Z = A_prev · W + b`, then ReLU on hidden layers and softmax on the output. Loss is categorical cross-entropy averaged over the batch.

For backprop, softmax combined with cross-entropy gives a simple output gradient, `dZ = (A - Y) / m`. Going backwards, `dW = A_prevᵀ · dZ`, `db = sum(dZ)`, and the error passes to the previous layer as `(dZ · Wᵀ) * ReLU'(Z_prev)`. Training is mini-batch gradient descent, reshuffling every epoch. Before training I checked the gradients against finite differences, and they agreed to about 1e-10.

## Experiments

Baseline: lr 0.1, batch 128, one hidden layer of 128, 15 epochs. Each run changes one thing.

| Run | Hidden layers | LR | Batch | Train acc | Val acc | Time (s) |
|---|---|---|---|---|---|---|
| lr=0.01 | 128 | 0.01 | 128 | 92.03 | 91.85 | 25 |
| lr=0.1 (baseline) | 128 | 0.1 | 128 | 97.88 | 96.28 | 21 |
| lr=0.5 | 128 | 0.5 | 128 | 99.87 | 97.35 | 27 |
| bs=32 | 128 | 0.1 | 32 | 99.74 | 97.07 | 40 |
| bs=512 | 128 | 0.1 | 512 | 94.53 | 93.71 | 21 |
| h=64 | 64 | 0.1 | 128 | 97.20 | 95.88 | 15 |
| h=256 | 256 | 0.1 | 128 | 98.12 | 96.45 | 42 |
| h=128-64 | 128, 64 | 0.1 | 128 | 99.21 | 96.71 | 27 |

## Results

lr=0.5 had the best validation accuracy, so I evaluated it on the held-out test split: 97.08% accuracy, with macro precision 0.9709, recall 0.9705 and F1 0.9706. That is 123 mistakes out of 4,209 images. Digit 6 and digit 1 scored best (F1 about 0.98), digit 9 worst (0.953). The most frequent confusions were 9 predicted as 7, 3 as 5, 9 as 4, 8 as 3, 4 as 9 and 3 as 2, each 6 to 7 times. <add one line on what you noticed in the misclassified-digit grid>

## Observations

The learning rate mattered most. At 0.01 the network was still far from converged after 15 epochs, and larger rates got much further in the same time. Batch size worked the same way: with 512 there are only a quarter as many updates per epoch as with 128, so it was undertrained, while 32 reached 97.07% but took about twice as long. In both cases the weaker result comes from the fixed 15-epoch budget, so these runs would probably catch up with more epochs.

The architecture changes moved accuracy by under 1.5 points. With about 4,200 validation images, one run's accuracy is uncertain by roughly 0.3 points, so lr=0.5, bs=32 and h=128-64 can't be ranked with confidence from a single seed. lr=0.5 also showed the largest train/validation gap (99.87% vs 97.35%), so it fits the training set closely and is the run most likely to overfit with longer training.

## What I learned

Softmax plus cross-entropy collapses to `A - Y` at the output, and the rest of backprop is the same pattern repeated layer by layer. Seeing that made the whole algorithm much less mysterious than the textbook chain-rule expansion. The finite-difference gradient check was worth doing: it's the only reliable way to know backprop is correct before a training run that would otherwise just fail to improve. Most of my debugging time with a from-scratch network would go to matrix shapes, so writing out every shape on paper first helps. And learning rate and the number of updates interact with the epoch budget, so comparing settings at a fixed number of epochs can make slow settings look worse than they are. Next I'd run several seeds per setting and add momentum or a learning-rate schedule.
