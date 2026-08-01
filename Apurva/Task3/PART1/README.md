# Task 3 - Part 1: K-Nearest Neighbors (KNN) Implementation & Analysis

Welcome to my repository for Part 1 of Task 3[span_0](start_span)[span_0](end_span)! In this project, I explored and implemented the K-Nearest Neighbors (KNN) classification algorithm using the Wine Quality dataset[span_1](start_span)[span_1](end_span). The main goal was to understand instance-based learning and figure out how data preprocessing, feature scaling, and hyperparameter tuning directly impact model performance[span_2](start_span)[span_2](end_span).

---

## 1. Workflow & Preprocessing Steps

Before training any models, I made sure the data was properly cleaned and prepared[span_3](start_span)[span_3](end_span):
* **Data Exploration:** Loaded the dataset to check its structure, summary statistics, and overall distributions[span_4](start_span)[span_4](end_span).
* **Missing Value Check:** Inspected the dataset for any null or missing values to ensure data integrity[span_5](start_span)[span_5](end_span).
* **Target Conversion:** Converted the original multi-class `quality` ratings into a clean binary classification task (`is_good_quality`) to make evaluation clearer.
* **Feature Scaling:** Tested three different scaling techniques—**No Scaling**, **StandardScaler**, and **MinMaxScaler**[span_6](start_span)[span_6](end_span)—to see how distance-based algorithms react to unscaled features. I found that `StandardScaler` yielded the most stable and reliable baseline performance.
* **Train-Test Split:** Split the dataset into 80% training and 20% testing sets using stratification to preserve class balance[span_7](start_span)[span_7](end_span).

---

## 2. Model Implementation & Hyperparameter Tuning

I ran a comprehensive grid search approach to evaluate different hyperparameter combinations:
* **K Values:** Tested various values of the $K$ hyperparameter ranging from 1 to 20[span_8](start_span)[span_8](end_span).
* **Distance Metrics:** Compared **Euclidean**, **Manhattan**, and **Minkowski** metrics[span_9](start_span)[span_9](end_span) to see how proximity is best measured across continuous chemical features.
* **Weighting Methods:** Evaluated both **Uniform** and **Distance** weighting schemes[span_10](start_span)[span_10](end_span) to test whether closer neighbors should hold more voting power.

---

## 3. Evaluation & Key Findings

To evaluate the models thoroughly, I generated performance plots for **Accuracy vs K** and **F1-Score vs K**[span_11](start_span)[span_11](end_span), alongside classification reports and confusion matrices. 

### Best Performing Configuration:
* **Optimal K Value:** 10
* **Optimal Distance Metric:** Euclidean
* **Optimal Weighting Method:** Distance
* **Achieved Accuracy:** ~88.15%
* **Achieved F1-Score:** ~87.76%

---

## 4. Conclusions & Justifications

1. **Why K = 10?** 
   A moderate value like $K=10$ provides the ideal balance between bias and variance. Very small values of $K$ (like 1 or 3) made the model overly sensitive to local noise, causing overfitting. Conversely, overly large values of $K$ oversmoothed the decision boundaries and missed subtle patterns for the minority class.
2. **Why Euclidean Distance & Standard Scaling?** 
   Since the dataset consists of continuous chemical properties (such as pH, alcohol, and acidity), standardizing the features ensures that each variable contributes equally to the geometric distance calculation. The Euclidean metric effectively captures straight-line similarity in this standardized continuous space.
3. **Why Distance-Based Weights?** 
   Using `weights='distance'` ensured that closer neighbors had a stronger voice than neighbors located further away. This significantly reduced misclassifications near overlapping class boundaries, yielding a cleaner and more accurate final decision boundary.
   