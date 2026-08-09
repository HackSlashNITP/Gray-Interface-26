# Part 3: Support Vector Machines (SVM) on Dry Bean Dataset

## AIM
The objective of this part for the task is to perform multi class classification on the dry bean dataset using Support Vector Machines (SVM) where the main aim was to compare four kernel types: Linear, Polynomial, Radial Basis Function, and Sigmoid , and evaluate model margins across accuracy, precision, recall, and F1-score.

---

## WORKFLOW
1. Exploratory Data Analysis:
   - Loaded Dry_Bean_Dataset.xlsx .
   - Identified 7 distinct bean varieties (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA).
   - Encoded categorical class names into numeric targets using LabelEncoder.

2. Feature Scaling:
   - Applied StandardScaler to ensure zero mean and unit variance across features.

3. Train-Test Split:
   - Here too , created an 80-20 stratified split to preserve bean class balance.

4. Kernel Comparison & Time Benchmarking:
   - Evaluated linear, poly, rbf, and sigmoid kernels.
   - Measured training time in seconds for each kernel.

5. Hyperparameter Tuning:
   - Tested regularization parameter for C in range [0.1, 1, 10].
   - Tested kernel coefficient for gamma in range ['scale', 0.01, 0.1].
   - Tested Polynomial degrees for [2, 3, 4].

---

## CONCLUSION
- RBF kernel consistently achieves top classification accuracy on non-linearly separable bean features.
- Linear kernels train fastest, while complex Polynomial and Sigmoid kernels take comparatively longer due to higher dimensional operations.
