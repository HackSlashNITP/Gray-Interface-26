# Task 2 - Part 3: Logistic Regression (Santander Customer Transaction Prediction)

## What I did
Loaded the Santander dataset from Kaggle (competition data)
Checked for any missing value (none found)
Checked target class distribution
Split the dataset in train and test (80/20)
Did scaling using StandardScale to scale all features on same scale
Trained Logistic Regression, experimented with different C values
Used GridSearchCV to find the best parameter
Evaluated using Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC

## Data Cleaning
Checked isnull().sum(), no missing values found

## EDA
Checked target value_counts() - class 0 has 179,902 rows and class 1 (transaction) has 20,098 rows roughly a 90/10 split. 
Clearly class imbalance so  used class_weight='balanced' in Logistic Regression to stop the model from just predicting the class 0 every time

Since all 200 features are unknown (var_0 to var_199) with no real names there wasn't anything to explore beyond checking the target imbalance

## Preprocessing
Did scaling using StandardScale as that's a standard choice for Logistic Regression

## Model Training
Logistic Regression finds a decision boundary by fitting coefficients that separate the two classes using a sigmoid function. Used class_weight='balanced' throughout to handle the class imbalance.

Tried different C values [0.001, 0.01, 0.1, 0.5, 1, 10, 50, 100] that controls regularization strength , smaller C = stronger regularization. Results remain almost same across all C values which implies regularization strength wasn't the deciding factor here  the class_weight balancing had a much bigger effect on the results than C did.

GridSearchCV over C, penalty (l1/l2/elasticnet), and max_iter to find the best combination. 
Best params found: C=0.001, max_iter=100, penalty='l2'.

## Model Evaluation
For different C Values
    C_Values Accuracy  Precision   Recall     F1
0    0.001  0.781650   0.292324  0.784936  0.425999
1    0.010  0.781225   0.291839  0.784694  0.425448
2    0.100  0.781275   0.291929  0.784936  0.425579
3    0.500  0.781225   0.291839  0.784694  0.425448
4    1.000  0.781225   0.291839  0.784694  0.425448
5   10.000  0.781225   0.291839  0.784694  0.425448
6  100.000  0.781225   0.291839  0.784694  0.425448


Overall Accuracy: 0.78

final model's Parameter : C=0.001, max_iter=100, penalty='l2'

from Confusion Matrix on final model 
TN (True Negative)  - 28025
FP (False Positive) - 7846
FN (False Negative) - 888
TP (True Positive) - 3241

## Observations
Accuracy (0.78) looks decent but is misleading here because of the class imbalance - a model predicting everything as class 0 would already get ~90% accuracy without being useful at all.

Precision for class 1 is low (0.29) , a lot of false positives (7846), meaning the model over-predicts the minority class.

Recall for class 1 is high (0.78) - the model does catch most actual positive cases, which is the point of using class_weight='balanced'.

ROC-AUC of 0.86 which means model is good at distinguishing b/w two classes overall, even though precision/recall on class 1 shows the tradeoff isn't perfect.


## Key Takeaway

Class imbalance was the main challenge in this dataset, not feature scaling or regularization. Using class_weight='balanced' had a much bigger impact on results than 'C'. The model catches most positive cases (good recall) but many false positives (low precision) 

## Notebook
Link to Kaggle Notebook: https://www.kaggle.com/code/anubhuti775/task-2-3
