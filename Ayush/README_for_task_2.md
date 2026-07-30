

# Gray Interface '26 – Task 2: Machine Learning Models

## Overview

This repository contains my solutions for **Gray Interface '26 – Task 2**, which focuses on implementing and evaluating core Machine Learning algorithms.
The objective of this assignment was to understand how regression and classification models work, how preprocessing affects model performance.

The assignment was divided into three independent notebook files, each covering a different machine learning task.

---

## Repository Structure

* **Part 1:** Linear Regression from Scratch using Gradient Descent
* **Part 2:** Linear Regression with Ridge, Lasso, and ElasticNet Regularization
* **Part 3:** Logistic Regression for Customer Transaction Prediction

---

# Part 1: Linear Regression from Scratch

### Workflow

* Generated a synthetic dataset containing 100 samples.
* Split the dataset into training and testing sets.
* Implemented Linear Regression manually using Gradient Descent without using `sklearn`'s `LinearRegression`.
* Experimented with different learning rates and numbers of epochs.
* Plotted the Loss vs Epoch graph to analyze convergence.
* Visualized the generated data and the fitted regression line.
* Implemented Polynomial Regression (Degree 2) as the bonus task and compared its performance with Linear Regression.

### Model Evaluation

The manually implemented model was evaluated using:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

Evaluation was performed on both the training and testing datasets.

### Key Observations

* Increasing the number of epochs reduced the loss until convergence.
* The learned regression coefficients closely approximated the original data-generating equation.
* Polynomial Regression produced similar performance because the underlying relationship in the generated dataset was approximately linear.

---

# Part 2: Linear Regression with Regularization

### Workflow

* Performed exploratory data analysis.
* Cleaned the dataset and handled missing values using suitable strategies.
* Created additional engineered features.
* Encoded categorical variables using Ordinal Encoding and One-Hot Encoding.
* Applied feature scaling using StandardScaler.
* Split the dataset into training and testing sets.

Instead of using `LinearRegression`, `SGDRegressor` was used as the baseline model as required in the assignment.

The following models were trained:

* SGDRegressor
* Ridge Regression
* Lasso Regression
* ElasticNet Regression

Cross-validation was used to determine the optimal regularization parameters.

### Model Evaluation

Each model was evaluated using:

* MAE
* MSE
* RMSE
* R² Score

Performance was compared on both training and testing datasets.

### Key Observations

* Feature scaling improved optimization and model stability.
* Ridge Regression reduced overfitting using L2 regularization.
* Lasso Regression performed feature selection by shrinking less important coefficients to zero.
* ElasticNet combined both L1 and L2 regularization.
* Among all models, Ridge Regression achieved the best overall balance between prediction accuracy and generalization on the testing dataset.

---

# Part 3: Logistic Regression

### Workflow

* Performed exploratory data analysis.
* Checked for and handled missing values.
* Applied feature scaling using StandardScaler.
* Split the dataset into training and testing sets.
* Trained a Logistic Regression classifier.
* Experimented with different values of the regularization parameter (C).
* Compared model performance for different regularization strengths.

### Model Evaluation

The classifier was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* ROC Curve

### Key Observations

* Feature scaling improved optimization and model convergence.
* Different values of C affected the balance between model complexity and generalization.
* The dataset exhibited class imbalance, making Precision, Recall, F1-score, and ROC-AUC more informative than Accuracy alone.
* The ROC Curve demonstrated that the model performed significantly better than random guessing.

---

# Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

# Conclusion

This assignment provided practical experience in implementing and evaluating both regression and classification models. It helped me to learn the importance of preprocessing,
feature engineering, feature scaling, regularization, hyperparameter tuning, and appropriate evaluation metrics. Through these experiments,
I developed a deeper and better understanding of how different machine learning models work or behave under various conditions and how preprocessing decisions influence overall model performance.

