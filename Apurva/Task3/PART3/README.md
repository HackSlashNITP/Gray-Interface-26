# Part 3: Support Vector Machines (SVM) - Dry Bean Dataset

## Overview
This part of the task explores Support Vector Machines (SVM) using the Dry Bean dataset[span_0](start_span)[span_0](end_span). The goal was to preprocess the data, test various kernels and hyperparameters, and analyze which configuration yields the best performance[span_1](start_span)[span_1](end_span).

## Workflow & Preprocessing
* **Dataset Loading:** Loaded the dataset from an `.arff` file format using `scipy.io` and converted it into a pandas DataFrame.
* **Preprocessing:** Checked for missing values, encoded target class labels into numeric format using `LabelEncoder`, and split the dataset into an 80-20 train-test split with stratification[span_2](start_span)[span_2](end_span).
* **Feature Scaling:** Applied `StandardScaler` to normalize features, ensuring optimal convergence for the SVM optimizer[span_3](start_span)[span_3](end_span).

## Model Training & Kernel Comparison
* Trained SVM models using four different kernels: **Linear, Polynomial, RBF, and Sigmoid**[span_4](start_span)[span_4](end_span).
* Recorded training times and evaluated each model using **Accuracy, Precision, Recall, and F1-Score**[span_5](start_span)[span_5](end_span).

## Hyperparameter Tuning
* Tested different configurations of regularization parameters ($C$), gamma ($\gamma$), and polynomial degrees to further optimize model accuracy[span_6](start_span)[span_6](end_span).

## Key Results & Conclusion
* **Best Performer:** The **RBF (Radial Basis Function) kernel** performed the best, achieving the highest accuracy ($~0.927$) and F1-score ($~0.938$) during hyperparameter tuning.
* **Why it works:** The Dry Bean dataset features complex, non-linear class boundaries. The RBF kernel handles these non-linear relationships much more effectively than linear or sigmoid options, as confirmed by a clean confusion matrix showing strong diagonal concentrations of correct predictions.
*