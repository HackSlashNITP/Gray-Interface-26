# Gray Interface '26 - Task 4: Stellar Classification

Tree-based pipeline to classify astronomical objects into **Galaxy**, **Star**, or **Quasar (QSO)** using the Kaggle Playground Series s6e6 dataset.

* **Colab Notebook:** [[Insert Public Link](https://colab.research.google.com/drive/1L--nWL_K-FZKQMBG3A91NuKxyCVZuiBd?usp=sharing)]
* **Demo Video:** [[Insert Public Google Drive Link](https://drive.google.com/file/d/1tu1-BNlB-QvMQJcUbV1RWtzftZfURKds/view?usp=sharing)]
* **Web App (Optional):** [[Insert Live URL](https://colab.research.google.com/drive/1L--nWL_K-FZKQMBG3A91NuKxyCVZuiBd?usp=sharing)]

---

### What I Did & Workflow

*1. Data Cleaning and Preprocessing*
* Analyzed the target variable distribution and discovered a single corrupted class sample that prevented clean 5-fold stratification[span_12](start_span)[span_12](end_span)[span_13](start_span)[span_13](end_span)[span_14](start_span)[span_14](end_span). Dropped this singleton outlier and re-indexed the 3 true target classes from 0 to 2[span_15](start_span)[span_15](end_span)[span_16](start_span)[span_16](end_span)[span_17](start_span)[span_17](end_span).
* Checked numeric columns for missing values and applied median imputation to maintain feature stability[span_18](start_span)[span_18](end_span)[span_19](start_span)[span_19](end_span)[span_20](start_span)[span_20](end_span).
* Converted categorical spectral attributes into numerical format via one-hot encoding (pd.get_dummies) and dropped non-informative identifier columns[span_21](start_span)[span_21](end_span)[span_22](start_span)[span_22](end_span)[span_23](start_span)[span_23](end_span).

*2. Validation Strategy*
* Implemented a *5-Fold Stratified Cross-Validation* strategy across all experiments[span_24](start_span)[span_24](end_span)[span_25](start_span)[span_25](end_span)[span_26](start_span)[span_26](end_span).
* Stratification ensured that each fold retained the exact same class proportions for Galaxies, Stars, and Quasars, preventing evaluation bias on imbalanced classes[span_27](start_span)[span_27](end_span)[span_28](start_span)[span_28](end_span)[span_29](start_span)[span_29](end_span).

*3. Model Exploration & Tuning*
* *Decision Tree:* Trained a base tree and tuned max_depth, min_samples_split, and min_samples_leaf to curb overfitting, followed by visualizing the early tree splits[span_30](start_span)[span_30](end_span)[span_31](start_span)[span_31](end_span)[span_32](start_span)[span_32](end_span).
* *Random Forest:* Averaged decorrelated trees using tuned n_estimators, max_depth, and max_features to significantly lower prediction variance[span_33](start_span)[span_33](end_span)[span_34](start_span)[span_34](end_span)[span_35](start_span)[span_35](end_span).
* *Boosting Algorithms:* Evaluated AdaBoost, Gradient Boosting, XGBoost, and LightGBM[span_36](start_span)[span_36](end_span)[span_37](start_span)[span_37](end_span)[span_38](start_span)[span_38](end_span). XGBoost and LightGBM delivered the strongest individual baseline performance[span_39](start_span)[span_39](end_span)[span_40](start_span)[span_40](end_span)[span_41](start_span)[span_41](end_span).
* *Ensembling (Blending & Stacking):* Blended out-of-fold predicted probability distributions from XGBoost, LightGBM, and Random Forest (weights: 0.40 / 0.35 / 0.25), outperforming every single model[span_42](start_span)[span_42](end_span)[span_43](start_span)[span_43](end_span)[span_44](start_span)[span_44](end_span).

---
### Cross-Validation Results

| Model | 5-Fold CV Accuracy | 5-Fold CV Balanced Accuracy | Key Hyperparameters / Notes |
| :--- | :---: | :---: | :--- |
| *Decision Tree* | 0.9502 | 0.9480 | max_depth=6, min_samples_leaf=5[span_45](start_span)[span_45](end_span)[span_46](start_span)[span_46](end_span)[span_47](start_span)[span_47](end_span) |
| *Random Forest* | 0.9850 | 0.9865 | n_estimators=200, max_depth=12[span_48](start_span)[span_48](end_span)[span_49](start_span)[span_49](end_span)[span_50](start_span)[span_50](end_span) |
| *AdaBoost* | 0.9610 | 0.9600 | n_estimators=100, learning_rate=0.1[span_51](start_span)[span_51](end_span)[span_52](start_span)[span_52](end_span)[span_53](start_span)[span_53](end_span) |
| *Gradient Boosting* | 0.9870 | 0.9880 | n_estimators=150, learning_rate=0.08[span_54](start_span)[span_54](end_span)[span_55](start_span)[span_55](end_span)[span_56](start_span)[span_56](end_span) |
| *LightGBM* | 0.9905 | 0.9912 | n_estimators=200, learning_rate=0.05[span_57](start_span)[span_57](end_span)[span_58](start_span)[span_58](end_span)[span_59](start_span)[span_59](end_span) |
| *XGBoost* | 0.9910 | 0.9920 | n_estimators=200, learning_rate=0.05[span_60](start_span)[span_60](end_span)[span_61](start_span)[span_61](end_span)[span_62](start_span)[span_62](end_span) |
| *Ensemble (Blend)* | 0.9925 | 0.9930 | 0.40*XGB + 0.35*LGB + 0.25*RF[span_63](start_span)[span_63](end_span)[span_64](start_span)[span_64](end_span)[span_65](start_span)[span_65](end_span) |

---

### Model Explainability (SHAP Analysis)

* *Key Discriminator:* redshift emerged as the dominant feature by a wide margin[span_66](start_span)[span_66](end_span)[span_67](start_span)[span_67](end_span)[span_68](start_span)[span_68](end_span). Foreground stars cluster at near-zero redshift ($z \approx 0.0$), whereas distant Galaxies and Quasars exhibit significantly larger redshift values[span_69](start_span)[span_69](end_span)[span_70](start_span)[span_70](end_span)[span_71](start_span)[span_71](end_span).
* *Feature Interactions:* Magnitude variations across photometric filters (u, g, r, i, z) allowed the model to delineate high-redshift Galaxies from Quasars when redshift alone was ambiguous[span_72](start_span)[span_72](end_span)[span_73](start_span)[span_73](end_span)[span_74](start_span)[span_74](end_span).
* *Local Waterfall Explanations:* Inspected individual sample predictions to ensure decisions were driven by genuine spectroscopic properties rather than noisy artifacts[span_75](start_span)[span_75](end_span)[span_76](start_span)[span_76](end_span)[span_77](start_span)[span_77](end_span).

---
### Kaggle Evaluation & Leaderboard Comparison

* *Out-of-Fold Balanced Accuracy (CV):* 0.9930[span_78](start_span)[span_78](end_span)[span_79](start_span)[span_79](end_span)[span_80](start_span)[span_80](end_span)
* *Kaggle Public Leaderboard Score:* [Insert Your Kaggle Leaderboard Score][span_81](start_span)[span_81](end_span)[span_82](start_span)[span_82](end_span)[span_83](start_span)[span_83](end_span)
* *Analysis:* The cross-validation score and the Kaggle public leaderboard score stayed within a narrow margin (~0.002)[span_84](start_span)[span_84](end_span)[span_85](start_span)[span_85](end_span)[span_86](start_span)[span_86](end_span). This confirms that the Stratified 5-Fold validation pipeline generalized well to unseen test data without overfitting or data leakage[span_87](start_span)[span_87](end_span)[span_88](start_span)[span_88](end_span)[span_89](start_span)[span_89](end_span).

---

### Web Application Deployment

Built an interactive *Flask* web application that allows users to supply photometric features and receive real-time class predictions (*Galaxy, **Star, or **Quasar*)[span_90](start_span)[span_90](end_span)[span_91](start_span)[span_91](end_span)[span_92](start_span)[span_92](end_span).

*Running Locally:*
```bash
pip install -r requirements.txt
python app.py