# Tree-Based Ensemble Learning Workflow (Task 4)

I built multiple tree-based machine learning models to classify astronomical objects into **GALAXY, STAR, and QSO** classes, I used the Kaggle Playground Series S6E6 dataset.

---

## 1. Exploratory Data Analysis (EDA)

The training dataset consists of **577,347 rows and 12 columns**, while the test dataset contains **247,435 rows and 11 columns**.

* ****Class Distribution:**** GALAXY is the majority class with 377,480 samples (~65.38%), followed by QSO with 117,143 samples (~20.29%) and STAR with 82,724 samples (~14.33%).

* ****Categorical Features:**** The dataset contains two categorical features, `spectral_type` and `galaxy_population`. The spectral types are `M`, `A/F`, `G/K`, and `O/B`.

* ****Feature Distributions:**** I used histograms and boxplots to visualize the numerical features such as `u`, `g`, `r`, `i`, `z`, and `redshift`.

* ****Outliers:**** I used the IQR method to identify outliers. Most outliers were found in the magnitude features and `redshift`. I did not remove the `redshift` outliers because they can represent genuine astronomical observations.

* ****Correlations:**** I found strong correlations between some magnitude features, especially `r-i` (~0.95) and `i-z` (~0.97). I kept these features because tree-based models can handle correlated features.

* ****Class-wise Analysis:**** I compared numerical and categorical features across GALAXY, STAR, and QSO classes using different visualizations.

---

## 2. Preprocessing Steps

* ****Missing Values:**** I checked missing values across all features and found **0 missing values**.

* ****Duplicate Values:**** I checked for duplicate rows and found **0 duplicate rows**.

* ****ID Removal:**** I removed the `id` column before training because it does not provide useful information for classification.

* ****Target Encoding:**** I encoded the target variable `class` using `LabelEncoder`.

* ****Categorical Encoding:**** I used `OneHotEncoder` to convert `spectral_type` and `galaxy_population` into numerical features.

* ****Data Splitting:**** I applied an 80/20 stratified train-validation split.

  * ****Train set:**** 461,877 samples

  * ****Validation set:**** 115,470 samples

---

## 3. Feature Engineering

I created four new features using differences between the magnitude bands:

* `u_g = u - g`
* `g_r = g - r`
* `r_i = r - i`
* `i_z = i - z`

These features provide additional information about the differences between the different magnitude bands.

The final dataset contained **12 numerical features and 2 categorical features**.

---

## 4. Model Implementation & Comparative Evaluation

I implemented different tree-based models and compared their performance using Accuracy, Balanced Accuracy, Precision, Recall, and F1-Score.

| Model             | Accuracy         | Balanced Accuracy | Precision        | Recall           | F1 Score         |
| :---------------- | :--------------- | :---------------- | :--------------- | :--------------- | :--------------- |
| ****XGBoost****   | ****0.958214**** | ****0.939095****  | ****0.957963**** | ****0.958214**** | ****0.958048**** |
| Random Forest     | 0.957331         | 0.937896          | 0.957080         | 0.957331         | 0.957168         |
| Decision Tree     | 0.937568         | 0.914547          | 0.937460         | 0.937568         | 0.937511         |
| Gradient Boosting | 0.931558         | 0.880642          | 0.930836         | 0.931558         | 0.929996         |
| AdaBoost          | 0.883589         | 0.866892          | 0.890159         | 0.883589         | 0.885847         |

****Observation:**** XGBoost achieved the highest validation performance. Random Forest also performed well, while AdaBoost and Gradient Boosting performed worse on this dataset.

---

## 5. Decision Tree

I implemented a Decision Tree classifier as the basic tree-based model.

I evaluated the model using Accuracy, Balanced Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

The Decision Tree achieved **93.76% accuracy** and **91.45% balanced accuracy**.

---

## 6. Random Forest

I implemented a Random Forest classifier using multiple decision trees.

The Random Forest achieved:

* ****Accuracy:**** 95.73%

* ****Balanced Accuracy:**** 93.79%

* ****F1-Score:**** 95.72%

Random Forest performed better than the single Decision Tree.

---

## 7. AdaBoost & Gradient Boosting

I also implemented AdaBoost and Gradient Boosting to compare different boosting methods.

* ****AdaBoost Accuracy:**** 88.36%

* ****Gradient Boosting Accuracy:**** 93.16%

Both models performed worse than Random Forest and XGBoost on this dataset.

---

## 8. XGBoost

I implemented XGBoost as the main gradient boosting model.

I used the following parameters:

* `n_estimators = 30`
* `max_depth = 8`
* `learning_rate = 0.1`
* `subsample = 0.8`
* `colsample_bytree = 0.8`

### XGBoost Performance

* ****Accuracy:**** 95.82%

* ****Balanced Accuracy:**** 93.91%

* ****Precision:**** 95.80%

* ****Recall:**** 95.82%

* ****F1-Score:**** 95.80%

XGBoost achieved the best validation performance and I selected it as the final model.

---

## 9. Soft Voting Ensemble

I developed a soft voting ensemble using **Decision Tree, Random Forest, and XGBoost**.

I compared the ensemble performance with the individual models.

The ensemble did not perform better than XGBoost, so I selected **XGBoost as the final model**.

---

## 10. SHAP Explainability

I used SHAP to understand how the XGBoost model makes predictions.

* I generated a SHAP feature importance plot.

* I generated a SHAP summary/beeswarm plot.

* I created dependence plots for the two most important features.

* I explained two individual predictions using SHAP force plots.

This helped me understand which features had the most influence on the model predictions.

---

## 11. Final Model & Kaggle Submission

I retrained the best XGBoost model using the complete training dataset and generated predictions for the test dataset.

I created the final `submission.csv` file with the required `id` and `class` columns.

The submission contained **247,435 predictions** and there were no missing values.

### Final Model Performance

* ****Validation Accuracy:**** `0.958214`

* ****Validation Balanced Accuracy:**** `0.939095`

* ****Validation F1-Score:**** `0.958048`

* ****Kaggle Leaderboard Score:**** `0.93977`

The Kaggle competition uses **Balanced Accuracy** as the evaluation metric.

---

## 12. Flask Web Application

I developed a simple Flask web application to make predictions using the XGBoost model.

The application takes the following inputs:

* `alpha`
* `delta`
* `u`
* `g`
* `r`
* `i`
* `z`
* `redshift`
* `spectral_type`
* `galaxy_population`

I also added the feature engineering steps inside the Flask application so that the required features are created automatically before prediction.

The trained model was saved as `best_model.pkl`.

The application predicts whether the astronomical object is **GALAXY, STAR, or QSO**.

---

## 13. Key Observations & Final Model Summary

* ****Best Model:**** `XGBoost`

* ****Validation Accuracy:**** `95.82%`

* ****Validation Balanced Accuracy:**** `93.91%`

* ****Validation F1-Score:**** `95.80%`

* ****Kaggle Score:**** `0.93977`

****Conclusion:****

I implemented different tree-based models for astronomical object classification. XGBoost achieved the best validation performance among the models I implemented, so I selected it as the final model. I also used SHAP to understand the model predictions and developed a simple Flask web application for making predictions using the trained model.
---

## 14. Demo Video

I recorded a short demo video showing the project workflow, model results, and the Flask web application.

**Demo Video:** [Task 4 Project Demo](https://drive.google.com/file/d/1EfawQ8gvs3yZqJJXER8UXBJF_W8ngEPE/view?usp=sharing)

---
