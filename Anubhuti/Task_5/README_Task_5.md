# Task 5: ANN from Scratch (MNIST Digit Classification)

1. Objective
Build and train a neural network from scratch using only NumPy to classify handwritten digits. The goal is to understand how forward propagation, loss, backpropagation and gradient descent actually work, without using TensorFlow, PyTorch or Keras.


2. Dataset overview
- Kaggle Digit Recognizer (MNIST): 42,000 labelled training images and 28,000 unlabelled test images.
- Each image is 28x28 grayscale = 784 pixel features, label is a digit 0-9.
- No missing values, duplicates found.
- Classes are fairly balanced (digit 1 slightly more, digit 5 slightly less), so accuracy is a fair metric, but precision, recall and F1 are also reported.


3. Pre-Processing & EDA
- Looked at pixel range (0-255), most pixels are 0 because digits sit in the middle of a black background.
- Plotted sample images of every digit, the average image of each class, and the class distribution.
- Normalised pixels to [0, 1] by dividing by 255.
- Shuffled and split into [~39,000] train / 1,000 validation / 2,000 held-out test.
- Validation set was used to compare experiments, held-out set only for the final evaluation.
- Kaggle 'test.csv' has no labels, so it was only used to visualise predictions.


4. ANN Architecture
- Input layer: 784
- Hidden layer: 128 neurons, ReLU
- Output layer: 10 neurons, Softmax
- Weights: He initialisation, biases start at 0
- Loss: categorical cross-entropy
- Optimiser: mini-batch gradient descent

Forward pass:
    Z1 = W1.X + b1,  A1 = ReLU(Z1),  Z2 = W2.A1 + b2,  A2 = softmax(Z2)

Backward pass:
    dZ2 = A2 - Y
    dW2 = (1/m) dZ2.A1^T,  db2 = (1/m) sum(dZ2)
    dZ1 = (W2^T.dZ2) * (Z1 > 0)
    dW1 = (1/m) dZ1.X^T,  db1 = (1/m) sum(dZ1)

'dZ2 = A2 - Y' comes from combining softmax with cross-entropy, the messy softmax derivative cancels out and the error is simply prediction minus target.


5. Training & Experiments
Mini-batch gradient descent was used, with train/validation loss and accuracy tracked every epoch. Experiments compared (all with [15] epochs):

Experiment      Learning rate   Batch size   Hidden units  Val acc  
Baseline        0.1             128          128            [0.9660]               
Low LR          0.01            128          128            [0.9080]              
Small batch     0.1             32           128            [0.9640]                
Full batch      0.1             ~39,000      128            [0.7300]              
Bigger hidden   0.1             128          256            [0.9650]               

(Training/validation loss and accuracy plots are in the notebook.)

- **Best configuration:** [ base lr=.1 bs=128 h=128] 


**Results** (held-out test set, 2,000 images)

- Accuracy: [0.9625]
- Macro Precision / Recall / F1: [0.9625] / [0.9621] / [0.9621]
- Per-class precision, recall, F1 and the confusion matrix are in the notebook.



6. Error Analysis
- Most confused digits: [e.g. 4 and 9, 3 and 5, 3 and 8] 
- Looking at the misclassified images: [e.g. messy strokes, digits that look like another class, model was low-confidence on many of them]


7. Observations & Conclusions

- Learning rate: [what happened with 0.01 vs 0.1]
- Batch size: [noise in curves, speed of convergence, how full batch compared]
- Hidden units: [did 256 help over 128, did 32 underfit]
- Overfitting: [gap between train and validation curves, if any]
- Best-performing setup: [ ]



9. Notebook
- 'NOTEBOOK' : https://www.kaggle.com/code/anubhuti775/ann-scratch-mnist
