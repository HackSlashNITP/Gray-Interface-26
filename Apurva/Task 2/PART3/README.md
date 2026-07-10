# Logistic Regression - Santander Customer Transaction Prediction

## About
In this project, I built a Logistic Regression model on the Santander Customer Transaction Prediction dataset. The main aim was to understand how Logistic Regression performs on a real-world classification problem and how preprocessing affects the final results.

## What I did
- Explored the dataset
- Checked for missing values
- Performed train-test split
- Applied feature scaling using StandardScaler
- Trained a Logistic Regression model
- Tried different values of the regularization parameter (C)
- Evaluated the model using Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, and ROC Curve

## Results
- Accuracy: **91.34%**
- Precision: **68.22%**
- Recall: **25.90%**
- F1-Score: **37.54%**
- ROC-AUC: **0.8599**

## My Observations
- There were no missing values in the dataset.
- Feature scaling improved the training process and is important for Logistic Regression.
- Changing the value of **C** from **0.01** to **100** did not make much difference in the model's performance.
- The dataset is imbalanced, so accuracy alone is not enough to judge the model.
- Even though the accuracy is above 91%, the recall is quite low, which means the model misses many positive cases.
- The ROC-AUC score of around **0.86** shows that the model is still able to separate the two classes fairly well.

## Conclusion
Overall, Logistic Regression gave a good baseline performance on this dataset. It achieved high accuracy and a good ROC-AUC score, but the low recall indicates that class imbalance affects the model. In the future, techniques like class balancing or more advanced models could be used to improve the prediction of the minority class.

## Libraries Used
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
