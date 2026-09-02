# Part 1: Classification using K-Nearest Neighbors (KNN)

## AIM:
to explore how K-Nearest Neighbors (KNN) works on tabular data, and to analyze how feature scaling, distance metrics, vote weighting strategies, and hyperparameter ($K$) selection impact model performance

---

## WORKFLOW: 
1. Exploratory Data Analysis (EDA) & Data Cleaning: 
   - Loaded the Wine Quality dataset containing red and white wine records.
   - Converted the categorical string feature type = (red/white) into numerical values (0 and 1).
   - Checked for missing values.

2. Train-Test Split:
   - Split the dataset into 80% training data and 20% testing data 

3. Feature Scaling Experimentation:
   - Tested baseline KNN (K=5) performance on Raw (Unscaled) data, StandardScaler, and MinMaxScaler.

4. Hyperparameter Tuning Loop:
   - Evaluated KNN models across odd values of K from 1 to 25.
   - Tested distance metrics: Euclidean, Manhattan, and Minkowski.
  

5. **Evaluation & Visualization**:
   - Plotted Accuracy vs K and F1-Score vs K.
  

---


## CONCLUSION:
- KNN calculates absolute distance between points. Without scaling, features with large scales dominate distance metrics completely.
- Manhattan distance worked better for this higher-dimensional dataset compared to Euclidean distance.
- wine quality ratings are imbalanced (most samples are rated 5 or 6, while 3, 4, 8, and 9 are rare), Weighted F1-Score was used as the true metric for tuning to ensure balanced predictive power.