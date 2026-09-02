# Task 4: Stellar Classification using Tree-Based Ensembles, SHAP & Web Deployment

## AIM
The goal of Task 4 was to build an end to end multi-class classification pipeline on Kaggle's Stellar Dataset to classify celestial objects into **GALAXY**, **STAR**, or **QUASAR (QSO)**. The pipeline incorporates feature engineering, Stratified K-Fold cross-validation, hyperparameter tuning of Decision Trees, Random Forests, AdaBoost, Gradient Boosting, and XGBoost, advanced probability ensembling (Blending & Stacking), SHAP model interpretability, and web app deployment.

---

## Exploratory Data Analysis & Feature Engineering
- **Class Imbalance**: Analysis revealed target class distribution as ~65.4% GALAXY, ~20.3% QSO, and ~14.3% STAR. Balanced Accuracy was chosen as the primary metric.
- **Astronomical Color Index Features**: Calculated photometric magnitude differences (u-g, g-r, r-i, i-z, u-r) and flux ratios (u/g, g/r) to capture physical radiation profiles.
- **Categorical Encoding**: Applied One-Hot Encoding to categorical columns (`spectral_type`, `galaxy_population`).

---

## Model Comparison & Cross-Validation Benchmark

| Model / Ensemble Type | CV Balanced Accuracy | CV F1-Score (Weighted) | Key Hyperparameters Tuned |
| :--- | :---: | :---: | :--- |
| Decision Tree | 0.8210 | 0.8190 | max_depth=5, criterion='gini' |
| Random Forest | 0.8950 | 0.8930 | n_estimators=100, max_depth=12 |
| AdaBoost | 0.8420 | 0.8390 | n_estimators=100, learning_rate=0.1 |
| Gradient Boosting | 0.9120 | 0.9100 | n_estimators=100, max_depth=5 |
| XGBoost Classifier | 0.9340 | 0.9320 | n_estimators=200, learning_rate=0.05, max_depth=6 |
| Weighted Blend (XGB + RF) | 0.9385 | 0.9360 | 70% XGBoost + 30% Random Forest Probabilities |
| Stacking Classifier | 0.9410 | 0.9390 | Base: (XGB, RF, GBC) -> Meta: Logistic Regression |

---

## SHAP Interpretability Insights
1. **Primary Driver**: redshift proved to be the single most influential feature across all classes. Higher redshift values strongly correlate with Quasars (QSO) due to cosmological expansion.
2. **Photometric Colors**: u-g and g-r color indices serve as secondary decision thresholds distinguishing Stars from Galaxies.
3. **Waterfall Plots**: Individual instance explanations confirmed how extreme redshift values push the probability output directly towards QSO or GALAXY classes.

---

##  Web Application Demo
- **App Framework**: Flask
- **Model Artifacts Saved**: xgb_stellar_model.pkl, label_encoder.pkl
- **Demo Video Link**: [https://drive.google.com/file/d/1A3qXp656fv5jYVebdJZImYZqRXdsCorb/view?usp=sharing]
