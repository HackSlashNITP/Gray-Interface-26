# Task 2 - Part 2: Linear Regression & Ridge/Lasso/ElasticNet (Ames Housing)

## What I did
Loaded the Ames Housing dataset from Kaggle.
Cleaned missing values 
Did Expolatory Data Analysis(EDA).
Encoded categorical columns with one-hot encoding.
Log-transformed the target (right-skewed), then standardized target and  features with Standard Scale.
Split the dataset in train and test  (80/20)
Trained Linear Regression, then 'SGDRegressor' as baseline.
Trained Ridge, Lasso, ElasticNet with cross-validation to have the best alpha.
Compared all models on MAE, MSE, RMSE, R2_score on both (train and test).

## Data Cleaning
Handle Missing Values :categorical feature with NaN filled with "None" where the feature doesn't exist for that house,
                      : numerical feature with NaNs filled with 0 or median depending on the column , checked either the house doesn't have the feature or data was not feeded.

Converted 'MS SubClass' to string data type as numeric looking code was actually representing building class categories

## EDA

Plotted heatmap to see top 10 most correlated numerical feature with saleprice , how they are related to each other that shoews Overall Qual, Gr Liv Area, and Garage Area were the features most strongly correlated with SalePrice.

Price rises almost linearly with living area and also with a few large-area/low-price outliers.

Also checked  Average SalePrice by Overall Quality, Sale Type, House Style, and Zoning (bar charts in the notebook) and  Overall Qual showed the clearest, most consistent trend.

## Encoding
Used one hot encoding for categorical features as these are unordered categories

## Target Transformation, Split & Scaling
considered saleprice column from dataset as Target 
checked skewness on target column and applied Log tranformation which fixes it 
splited the dataset train,test(80/20)
Did scaling to scale all feature on same scale using 'StandardScale' as standard choice in case of linear\regularisation 

## Model Training

LinearRegression solves for exact optimal weights in one step . 'SGDRegressor' updates weights gradually by gradient descent  Needed careful tuning here: my first attempt diverged (huge negative R2), and only worked after switching to learning_rate='invscaling' with a smaller eta0.
Ridge shrinks coefficients toward zero using L2 penalty (keeps all features, reduces their impact). Lasso uses L1 penalty, which can zero out coefficients entirely (effectively removes unimportant features). ElasticNet mixes both.

## Best Regularization Parameters (from CV)

Ridge with  best alpha = 100 
Lasso with  best alpha = 0.01 
ElasticNet with  best alpha = 0.01, l1_ratio = 0.5


## Model comparison

               Model  Train MAE  Train RMSE  Train R2  Train MSE   Test MSE  \
0  Linear Regression   0.163017    0.244256  0.940339    0.059661  0.190873   
1      SGD Regressor   0.226464    0.337215  0.886286    0.113714  0.098392   
2              Ridge   0.175051    0.262427  0.931132    0.068868  0.108346   
3              Lasso   0.193187    0.291360  0.915109    0.084891  0.104819   
4         ElasticNet   0.184984    0.278313  0.922542    0.077458  0.107047   

   Test MAE  Test RMSE   Test R2  
0  0.222547   0.436890  0.796783  
1  0.231315   0.313675  0.895245  
2  0.215712   0.329160  0.884647  
3  0.211783   0.323758  0.888402  
4  0.210447   0.327181  0.886030 

*copied from the notebook


## Observations from Model Comparison
Linear Regression had the highest R2 on train (0.940) but the lowest R2 on test (0.797) , case of overfitting.
SGD, Ridge, Lasso, and ElasticNet all  havinf almost same R2 on test (0.885 - 0.895)
SGD had the best test R2 (0.895) among all models, though the margin over the regularized models is small.


## Key Takeaway
Regularization actually helped here. Every regularized model (Ridge, Lasso, ElasticNet) and SGD did better than just Linear Regression on the test data even though Linear Regression fit the training data the best. This shows Linear Regression was overfitting and the regularized models generalized better to new data. SGD also needed a lot of care with the learning rate to actually converge, unlike Linear Regressions.

## Notebook
Link to Colab Notebook: https://colab.research.google.com/drive/1Gxg8N-ZMd5jjoT5TiTCLu2jq4a-8RtuE?usp=sharing
