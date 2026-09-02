# Dry Bean Dataset Classification using Support Vector Machines (SVM)

## Overview
This notebook demonstrates a complete machine learning pipeline for classifying different types of dry beans using the 'Dry Bean Dataset' from Kaggle. The process involves data loading, exploratory data analysis (EDA), data preprocessing, training various Support Vector Machine (SVM) models with different kernels and hyperparameters, and evaluating their performance.

## Dataset
The 'Dry Bean Dataset' contains images of 13,611 grains of 7 different registered dry beans, which were extracted from the samples and subjected to 16 features. The dataset aims to classify these bean types based on their morphological properties.

**Source:** [Kaggle: Dry Bean Dataset](https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset)

## Notebook Contents

1.  **Data Loading:**
    *   The dataset is downloaded from Kaggle and loaded into a pandas DataFrame using `pd.read_excel`.

2.  **Exploratory Data Analysis (EDA):**
    *   **Dataset Information:** `df.info()` is used to inspect data types and non-null counts.
    *   **Missing Values Check:** `df.isnull().sum()` confirms the absence of missing data.
    *   **Descriptive Statistics:** `df.describe()` provides statistical summaries of numerical features.
    *   **Target Variable Distribution:** A count plot visualizes the distribution of the 7 different bean classes (`Class` column).

3.  **Data Preprocessing:**
    *   **Feature-Target Split:** The dataset is split into features (`X`) and the target variable (`y`, which is 'Class').
    *   **Train-Test Split:** The data is divided into training (70%) and testing (30%) sets using `train_test_split` with `stratify=y` to maintain class distribution.
    *   **Feature Scaling:** `StandardScaler` is applied to normalize the features, which is crucial for SVM performance.

4.  **SVM Model Training and Evaluation:**
    *   **Kernels Explored:** Linear, Polynomial, RBF (Radial Basis Function), and Sigmoid kernels are used.
    *   **Hyperparameter Tuning:**
        *   `C_values`: `[0.1, 1, 10]`
        *   `gamma_values`: `['scale', 'auto', 0.1, 1]`
        *   `degree_values`: `[2, 3, 4]` (for the Polynomial kernel only)
    *   **Performance Metrics:** For each model configuration, the following metrics are recorded:
        *   Training Time
        *   Accuracy
        *   Precision (weighted)
        *   Recall (weighted)
        *   F1-Score (weighted)
        *   Confusion Matrix

## Key Findings

After training and evaluating numerous SVM models with different kernel types and hyperparameter combinations, the following model was identified as the **best performer based on accuracy**:

*   **Kernel:** RBF
*   **C:** 10.0
*   **Gamma:** 0.1
*   **Accuracy:** Approximately 92.80%
*   **Precision (weighted):** Approximately 92.76%
*   **Recall (weighted):** Approximately 92.80%
*   **F1-Score (weighted):** Approximately 92.76%

The RBF kernel, with its ability to handle non-linear decision boundaries, demonstrated superior performance on this dataset when tuned with `C=10.0` and `gamma=0.1`. This combination likely allowed the model to effectively capture the complex relationships within the bean features without overfitting.

The notebook outputs include a detailed DataFrame of all model results and a confusion matrix for the best-performing model, providing a comprehensive overview of the classification performance.