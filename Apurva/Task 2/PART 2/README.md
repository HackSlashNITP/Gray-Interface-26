# Ames Housing Price Prediction using Linear Regression and Regularization

## Overview

This project focuses on predicting house prices using the Ames Housing dataset. Along with a baseline linear regression model, different regularization techniques were applied to understand their effect on model performance and generalization.

## Workflow

The project was completed in the following order:

- Loaded and explored the dataset.
- Checked the dataset structure and identified missing values.
- Separated numerical and categorical features.
- Filled missing values using appropriate imputation methods.
- Applied one-hot encoding to categorical features.
- Standardized numerical features.
- Split the dataset into training and testing sets.
- Trained and evaluated four regression models.
- Compared their performance using multiple evaluation metrics.

## Preprocessing

The following preprocessing steps were performed before model training:

- Missing numerical values were filled using the median.
- Missing categorical values were filled using the most frequent value.
- Categorical variables were converted into numerical form using One-Hot Encoding.
- Numerical features were standardized using StandardScaler.
- An 80:20 train-test split was used for model evaluation.

## Models Used

The following models were implemented:

- SGDRegressor (used as the baseline linear regression model)
- Ridge Regression
- Lasso Regression
- Elastic Net Regression

Cross-validation was used for selecting the regularization parameters of the regularized models.

## Evaluation

Each model was evaluated on both the training and testing datasets using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

These metrics were used to compare prediction accuracy and the ability of each model to generalize to unseen data.

## Observations

- The baseline SGDRegressor showed poor performance compared to the regularized models.
- Ridge Regression produced strong and stable results.
- Lasso Regression achieved the best overall performance with the highest test R² score and the lowest prediction errors.
- Elastic Net also performed well but was slightly behind Lasso on this dataset.

## Conclusion

Among all the models, Lasso Regression performed the best. It achieved the lowest prediction errors and the highest test R² score, making it the most suitable model for predicting house prices on the Ames Housing dataset. The comparison also highlighted the importance of regularization in improving model performance over the baseline linear regression approach.