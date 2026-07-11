# LOGISTIC REGRESSION

## Aim of this task : 
The goal of this task was to predict whether a customer would complete a transaction i.e, target = 1 or not that is target = 0 using the provided dataset. 
Handled a severe class imbalance (90% zeros vs 10% ones)and experimented with the inverse regularization parameter(C).

## Workflow: 
Did the following steps to complete this one: 

##### Stratified Sampling & Splitting
Due to the large size of the dataset- 200,000 rows, i extracted a stratified sample of 40,000 rows to speed up training while maintaining the 90/10 class split. then further split this sample into an 80% training set and a 20% testing set.

##### Feature Scaling
Applied StandardScaler to clean the numerical features , ensured stable gradient descent convergence and fair regularization penalties.

##### Handling Class Imbalance
Because 90% of the data consists of customers who didn't buy anything. I tried fixing this by turning on class_weight='balanced', which forces the model to pay extra attention to the 10% of customers who actually made a transaction.

##### Regularization Tuning (C)
Loop tested four different C that is - [C = 0.001, 0.1, 1, 100] and collected all evaluation metrics inside a clean Pandas comparison DataFrame.

## Observation:
Changing the C value made almost no difference to our results. Because of this, C = 0.001 is our best choice because it keeps the model simple and fast without losing any accuracy.



