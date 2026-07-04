# Gray Interface '26 — Task 2: Regression & Classification

## Notebook Links

| Part | Colab Link | Dataset |
|---|---|---|
| Part 1 — Linear Regression from Scratch | [Open in Colab](https://colab.research.google.com/drive/1Pdht-Eqk-jhIekx67bqMIhZIAu40bWgr?usp=sharing) | Synthetic (generated in notebook) |
| Part 2 — Ridge / Lasso / ElasticNet | [Open in Colab](https://colab.research.google.com/drive/1AB677lW3R3Y9jsMQ1Na1cxrtSy95CfqZ?usp=sharing) | [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset?utm_source=chatgpt.com) |
| Part 3 — Logistic Regression | [Open in Colab](https://colab.research.google.com/drive/1AHSkUhvIoQEmaaz6Bzpc3jLcXcd6kfJI?usp=sharing) | [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) |

---

## Part 1 — Linear Regression from Scratch

### Steps Performed
1. Generated a synthetic 100-sample dataset with a known linear relationship.
2. Visualized the data with a scatter plot to confirm the linear trend before modeling.
3. Split the data into training (80%) and testing (20%) sets.
4. Implemented **Gradient Descent** manually to learn the slope (`m`) and intercept (`b`) —
   without using `sklearn.LinearRegression`.
5. Fitted the model and plotted the resulting regression line over the training data.
6. Experimented with different learning rates (0.001, 0.01, 0.1) and epoch counts
   (100, 1000, 10000) to study convergence behavior.
7. Plotted **Cost vs Epochs** to visualize how the loss decreases during training.
8. Evaluated the model using MAE, MSE, RMSE, and R² on both train and test sets.

### Concept Behind Gradient Descent
Gradient Descent minimizes the Mean Squared Error by repeatedly moving `m` and `b` in the
direction opposite to the loss gradient, scaled by the learning rate.

### Curve Details
**Cost vs Epochs:** The curve drops sharply at first and then flattens — the flattening
marks convergence (the point where further training gives diminishing returns).
- Very low learning rate + few epochs → curve hasn't flattened yet → poor fit.
- Higher learning rates reach the flat region in fewer epochs.
- Once flattened, increasing epochs further gives no meaningful improvement.

### Key Observations
- Learned parameters (m ≈ 3.42, b ≈ 3.54) closely matched the true underlying relationship
  used to generate the data.
- Train and test metrics (R² ≈ 0.74 on both) were very close, showing good generalization
  with no over/underfitting.
- Learning rate and epochs trade off against each other: a learning rate too low needs far
  more epochs to converge, while a learning rate too high converges fast but risks instability
  on more complex/less-scaled data.
- **Best configuration:** L = 0.01 with 100 epochs — reached the same performance (R² ≈ 0.74)
  as slower configurations but with far fewer iterations.

---

## Part 2 — Linear Regression & Ridge / Lasso / ElasticNet (Ames Housing)

### Steps Performed
1. Loaded the Ames Housing dataset and inspected its structure, data types, and missing values.
2. Plotted the distribution of `SalePrice` to check skewness before modeling.
3. Applied a **log transformation** to `SalePrice` — regression assumes roughly normal,
   evenly spread residuals, and raw prices are right-skewed, so log-scaling makes the
   target better suited to a linear model.
4. Removed outliers from the log-transformed target using the **IQR method**, so a small
   number of extreme prices don't distort the fitted coefficients.
5. Built a preprocessing pipeline: missing numeric values filled with the median, missing
   categorical values filled with a constant label, numeric features standardized, and
   categorical features one-hot encoded — all combined so the same transformation is
   applied consistently to train and test data.
6. Trained a **Linear Regression** model as a baseline, then an **SGDRegressor** as an
   alternative baseline (an iterative, gradient-based solver instead of the closed-form solution
   used by Linear Regression — better suited to very large datasets, but more sensitive to
   tuning).
7. Trained **Ridge**, **Lasso**, and **ElasticNet** to study the effect of regularization.
8. Evaluated every model with MAE, MSE, RMSE, and R² on both train and test data.
9. Plotted **Actual vs Predicted** values for each model.

### Concept Behind Regularization
- **Ridge (L2):** shrinks coefficients smoothly, reducing variance without removing features —
  useful when many correlated/dummy features exist.
- **Lasso (L1):** can shrink coefficients to exactly zero, performing feature selection —
  but too strong a penalty can remove genuinely useful features and underfit.
- **ElasticNet:** blends L1 and L2 to balance feature selection with coefficient shrinkage.

### Curve Details
**Actual vs Predicted (with a y = x reference line):** points closer to the diagonal line
mean more accurate predictions.
- Linear Regression and Ridge showed points tightly clustered near the diagonal.
- Lasso and SGDRegressor showed noticeably wider scatter away from the diagonal.

### Key Observations
- **Ridge Regression performed best overall** — highest test R² and lowest test RMSE, with
  a smaller gap between train and test performance than plain Linear Regression, meaning it
  generalized slightly better.
- Lasso underperformed on both train and test data — the regularization strength used was
  likely too high for a dataset with many one-hot encoded columns, zeroing out useful features
  and causing underfitting.
- ElasticNet performed between Ridge and Lasso, as expected from blending both penalties.
- SGDRegressor underperformed the closed-form models, showing that iterative solvers need
  careful tuning of learning rate and iterations to match an exact solution on data of this size.

---

## Part 3 — Logistic Regression (Classification)

### Steps Performed
1. Loaded the Santander dataset and checked its structure and missing values.
2. Built a preprocessing pipeline (median imputation and scaling for numeric features,
   constant imputation and one-hot encoding for categorical features).
3. Trained a **Logistic Regression** model and generated predictions.
4. Evaluated the model using Accuracy, Precision, Recall, and F1-score on both train and
   test sets.
5. Experimented with different values of the regularization parameter **C** to observe its
   effect on performance.
6. Plotted the **ROC Curve** to visualize classification performance across thresholds.
7. Computed the **Confusion Matrix** for both train and test predictions.

### Concept Behind Key Ideas
- **Feature scaling** matters for Logistic Regression because it is a gradient-based linear
  model — unscaled features with larger ranges would dominate the learned coefficients.
- **Regularization parameter C** is the inverse of regularization strength: a smaller C means
  stronger regularization (simpler decision boundary), while a larger C fits the training
  data more closely.
- **Confusion Matrix** breaks predictions into true/false positives and negatives, forming
  the basis for precision, recall, and F1.

### Curve Details
**ROC Curve:** plots True Positive Rate against False Positive Rate across thresholds; the
diagonal line represents random guessing (AUC = 0.5). Both the training and testing curves
bowed towards the top-left corner, away from the diagonal, indicating the model separates
the two classes well above chance, with train and test curves overlapping closely.

### Key Observations
- Performance stayed nearly flat across C values from 0.1 to 100, meaning regularization
  strength had little effect — the informative features were separable enough that the
  model wasn't overfitting even with weak regularization.
- The strongest regularization (C = 0.01) gave marginally the best accuracy, suggesting a
  very simple decision boundary already captures the signal well.
- Train and test performance were close to each other, indicating good generalization with
  no significant overfitting.
- The confusion matrices showed a fairly balanced number of false positives and false
  negatives across both classes, consistent with a balanced class distribution.
