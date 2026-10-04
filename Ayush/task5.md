# Gray Interface '26 — Task 5
## Artificial Neural Network (ANN) from Scratch

### Objective
I implemented an Artificial Neural Network (ANN) from scratch using NumPy to classify handwritten digits from the MNIST dataset. The main objective was to understand how neural networks learn through forward propagation, loss calculation, backpropagation, and gradient descent without using deep learning frameworks.

### Dataset
I used the **Kaggle Digit Recognizer** dataset, which contains:
- **Training data:** 42,000 labeled images
- **Test data:** 28,000 unlabeled images
- **Image dimensions:** 28 × 28 pixels
- **Input features:** 784 pixels per image
- **Output classes:** 10 digits (0–9)

### Exploratory Data Analysis (EDA)
I performed exploratory data analysis to understand the dataset:
- Visualized sample handwritten digit images.
- Analyzed the distribution of digits.
- Checked for missing values and duplicate records.
- Normalized pixel values to the range [0, 1].
- Converted digit labels into one-hot encoded vectors.
- Split the training data into 80% training and 20% validation sets using stratified sampling.

### Model Architecture
I built the ANN entirely using **NumPy**, implementing the core neural network operations manually.

- **Input layer:** 784 neurons
- **Hidden layers:** ReLU activation
- **Output layer:** 10 neurons with Softmax activation
- **Loss function:** Categorical Cross-Entropy
- **Optimization:** Mini-batch Gradient Descent

I implemented forward propagation, loss calculation, backpropagation, and weight and bias updates from scratch.

### Experiments and Results
I conducted three experiments with different network architectures and hyperparameters to compare their performance.

| Experiment | Hidden Layers | Learning Rate | Batch Size | Epochs | Validation Accuracy |
|---|---|---:|---:|---:|---:|
| Experiment 1 | [64] | 0.01 | 64 | 15 | 92.45% |
| Experiment 2 | [128, 64] | 0.01 | 64 | 15 | 94.61% |
| Experiment 3 | [256, 128] | 0.005 | 128 | 15 | 91.52% |

**Best Model:** Experiment 2, with hidden layers [128, 64], achieved the highest validation accuracy of **94.61%**.

### Model Evaluation
I evaluated the best-performing model using the following metrics:

| Metric | Score |
|---|---:|
| Accuracy | 94.61% |
| Macro Precision | 94.60% |
| Macro Recall | 94.52% |
| Macro F1-Score | 94.55% |

I also generated a confusion matrix to analyze classification performance across different digits and visualized misclassified images to understand the model's errors.

### Observations
- Experiment 2 achieved the best validation accuracy among the three experiments.
- Increasing the number of neurons and hidden layers did not necessarily improve performance.
- The confusion matrix helped identify digits that were frequently misclassified.
- Training and validation curves were used to observe the model's learning behavior and generalization.

### Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

### Conclusion
I successfully implemented and trained an Artificial Neural Network from scratch using NumPy for handwritten digit classification. The best model achieved a validation accuracy of **94.61%** and a macro F1-score of **94.55%**.

Through this task, I gained practical understanding of neural network architecture, forward propagation, backpropagation, mini-batch gradient descent, and model evaluation without relying on deep learning frameworks.