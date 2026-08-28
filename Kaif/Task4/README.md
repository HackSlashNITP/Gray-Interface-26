# Stellar Object Classifier

Classifies astronomical objects photometric and spectral features into **Galaxy**, **QSO (Quasar)**, or **Star**.

## Notebook
[Google Colab Notebook](https://colab.research.google.com/drive/1Cn0UhOKylgEVbsXtwyM9_z84nmG0V8Op?usp=sharing)

## EDA & Preprocessing
- Checked class distribution, missing values, and duplicates.
- Analyzed numerical features with histograms and boxplots; detected outliers using the IQR method.
- Checked feature correlations.
- Preprocessing pipeline (`ColumnTransformer`): median imputation + `StandardScaler` for numeric features, most-frequent imputation + `OneHotEncoder` for categorical features.
- Target (`class`) encoded with `LabelEncoder`.

## Models Trained
- Decision Tree (baseline)
- Random Forest
- AdaBoost
- Gradient Boosting
- XGBoost
- **Stacking Ensemble** — base learners: XGBoost, AdaBoost, Gradient Boosting; meta-learner: Random Forest

Each model was tuned via `GridSearchCV`/hyperparameter search (`n_estimators`, `learning_rate`, `max_depth`,`min_samples_leaf` ) and evaluated with **Stratified 5-Fold Cross-Validation**.

## Model Comparison
Models were compared on Accuracy and Balanced Accuracy; the Stacking Ensemble was evaluated against the best individual model (XGBoost).

## Feature Importance & Explainability (SHAP)
Applied SHAP to the best-performing tree-based model (XGBoost):
- SHAP feature importance (bar) plot
- SHAP summary/beeswarm plot
- Dependence plots for the top 2 features
- Waterfall plots explaining 2 individual predictions

`redshift` was consistently the most influential feature, consistent with the physics of quasar/galaxy classification.

## Kaggle Submission
- Generated predictions on `test.csv` using the final model and submitted to Kaggle.
- Cross-Validation Accuracy: 0.9637 (± 0.0003)
- Kaggle Public Leaderboard Score: <0.95296>

## Web Application
- **Backend:** FastAPI, serves the saved pipeline (`joblib`) via a `/predict` endpoint.
- **Frontend:** Streamlit, collects user input and calls the FastAPI backend.
- **Backend hosted on:** Render (free tier)
- **Frontend hosted on:** Streamlit Community Cloud

**Live app:** https://stellar-object-classifier-2muuzp2unzprtbyvqrwfhb.streamlit.app/

**Demo video:** https://drive.google.com/file/d/10PCTNGsnG5xF9jn5_EI6iZG04JJ9Ez9L/view?usp=sharing

