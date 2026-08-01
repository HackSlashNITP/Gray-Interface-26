#  Linear Regression and Ridge/Lasso/ElasticNet Regularization

## Aim of this task :
Here, the main objective we had was to apply data preprocessing techniques and test different types of regularization (Ridge, Lasso, Elastic Net) to prevent overfitting and make accurate predictions on unseen data.

## Workflow :
Tried completing this task in the following four steps: 

#### STEP-1 : Data cleaning and splitting
Separated the features from the target variable (SalePrice) and dropped identifiers (Order, PID).
For splitting - followed the general trend, of spliting the data into an 80% training set and a 20% testing set. This splitting was done before doing any scaling to prevent data leakage.

#### STEP- 2: Pipeline Preprocessing
processed columns based on data type -

For Numerical Columns , filled missing values with the column median and scaled features using StandardScaler.

For Categorical Columns , filled missing text fields with the string 'Missing' and converted text to binary numbers using OneHotEncoder.

#### STEP-3: Parameter Tuning
Used 5-Fold Cross-Validation to automatically find the best penalty strength(alpha) for the models.

#### STEP-4: Model Comparison
Trained four different models that includes : Vanilla Linear Regression, Ridge, Lasso, and Elastic Net using Scikit-Learn’s SGDRegressor and evaluated them inside a clean Pandas DataFrame.

## Observations :
Lasso Regression performed the best, matching Vanilla Regression with a strong Test $R^2$ score of 0.8945 and the lowest Test RMSE ($29,083.05).
