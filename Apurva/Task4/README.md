# Gray Interface '26 - Task 4: Stellar Classification

Tree-based pipeline to classify astronomical objects into **Galaxy**, **Star**, or **Quasar (QSO)** using the Kaggle Playground Series s6e6 dataset.

* **Colab Notebook:** [[Insert Public Link](https://colab.research.google.com/drive/1L--nWL_K-FZKQMBG3A91NuKxyCVZuiBd?usp=sharing)]
* **Demo Video:** [[Insert Public Google Drive Link](https://drive.google.com/file/d/1tu1-BNlB-QvMQJcUbV1RWtzftZfURKds/view?usp=sharing)]

---

### What I Did & Workflow

1. Project Overview & Workflow
The goal was to build an end-to-end classification pipeline for stellar object identification (Galaxy, Star, Quasar) using data from Kaggle Playground Series s6e6.  
My approach:
EDA & Preprocessing: Explored class balances, handled nulls/outliers, and scaled/engineered relevant features.  
Validation Setup: Used 5-Fold Stratified K-Fold cross-validation to preserve class distributions across splits.  
Model Exploration: Baseline Decision Tree \rightarrow Random Forest \rightarrow Gradient Boosting / XGBoost \rightarrow Stacking/Blending.  
Explainability: Evaluated model decisions using SHAP beeswarm, feature dependence, and local prediction force plots.  
Deployment: Exported the top model and built an interactive web app with Flask/FastAPI.

2. Preprocessing & Feature Engineering
Data Cleaning: Checked missing values, removed duplicates, and inspected extreme outliers in the feature distributions.  
Feature Transformations: Handled skewed numeric features and dropped redundant/collinear columns identified during correlation analysis.  
Feature Importance: Evaluated preliminary tree feature importances to weed out uninformative features early.

3. Model Training & Comparison
I evaluated models primarily on Balanced Accuracy and F1-Score to account for any class distribution imbalance.
Key Takeaways:
The standalone Decision Tree was prone to overfitting on deeper splits; pruning with max_depth and min_samples_leaf helped, but Random Forest consistently outperformed it.  
XGBoost provided the strongest single-model accuracy, and blending it with Random Forest yielded the most stable out-of-fold generalization.

4. Model Interpretability (SHAP Analysis)
I ran SHAP on the best-performing model to understand feature impact:
Global Importance & Summary Plot: Verified which spectral features drive class decisions most heavily across the dataset.  
Dependence Plots: Analyzed how variations in top numerical features directly alter model output probabilities.  
Local Explanations: Inspected 2 individual sample predictions to verify how specific feature values pushed the prediction toward a given stellar class.

Running the App Locally
# Clone the repository and navigate to the directory
cd Apurva

# Install requirements
pip install -r requirements.txt

# Run the server
python app.py