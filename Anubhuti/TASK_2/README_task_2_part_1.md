# Task 2_Part 1: Linear Regression from Scratch (Gradient Descent)

## What I did
Generated a synthetic dataset (100 samples, 1 feature) using 'make_regression'
Splited into train and test (80:20).
Applied Linear Regression from scratch using Gradient Descent(without using skitlearn)
Fited the model and visualized the regression line on scatter plot.
Experiment with different learning rates and epochs.
Plotted  different Loss vs Epoch to check convergence.
Evaluated with MAE, MSE, RMSE, R2_score on train and test datasets.
Improved the previous model to support multiple feature , 'MyGDRegressorMulti'
Bonus part : Implemented Polynomial Regression (degree 2).

## Learning Rate Experiments
5 model at different learning rate with 100 epochs each all converge with this epochs except learning rate with 0.001 needed more epochs 

Observation: lr=0.001 was too slow with 100 epochs , it starts converges after 2000 epochs . Ecperimented with lr=(0.01 - 0.1) all converged to almost same loss around (656) means this dataset tolerates wide range of learning rates without diverging. Earlier I mistakenly saw signs of divergence at higher learning rates,it was because there was a bug in my gradient formula (I was using 'sum()' instead of'mean()` for the gradient terms). After fixing the gradient to properly average over samples, the model converges reliably across all tested learning rates.

## Final Model Performance (lr=0.01, 1000 epochs)
MAE  -- train(20.53)  --- test(25.25)
MSE  -- train(656.30) ---test(937.82)
RMSE  -- train(25.62) ---test(30.62)
R2_score -- train(0.751) -- test(0.654)


## Bonus: Polynomial Regression (Degree 2) Vs Linear regression (Test Dataset)
MAE  --- linear(25.25) and polynomial(26.71)
MSE  --- linear(937.82) and polynomial (1132.39)
RMSE --- linear(30.62) and for polynomial(33.65) 
R2_score --linear(0.654) and for polynomial(0.582) 

Observation: Polynomial regression performed slightly worse than linear regression on both train and test sets as the synthetic dataset was generated with linear relationship, adding a quadratic term doesn't help  it just adds unnecessary complexity, which slightly hurts generalization.

## Key Takeaway
Gradient descent's stability depends heavily on how the gradient is scaled - normalizing by sample size (mean, not sum) makes convergence much more robust across learning rates. For this dataset, learning rates between 0.01-0.1 converge reliably, smaller rates need more epochs to reach the same point. Model complexity (polynomial vs linear) should match the true structure of the data adding degree didn't help since the data was linearly generated.

## Notebook

Link to collab Notebook : https://colab.research.google.com/drive/1Gmk9P3lSQ9BkebY0Ry5-pMiD87M48YPS?usp=sharing
