# Gray Interface '26 — Task 3

This folder contains my notebooks for Task 3 (Classification).

## Notebooks

| File | Part | Algorithm | Dataset |
|------|------|-----------|---------|
| `Task3_Part1_KNN.ipynb` | Part 1 | K-Nearest Neighbors | Wine Quality (Red & White) |
| `Task3_Part2_NaiveBayes.ipynb` | Part 2 | Multinomial Naive Bayes | SMS Spam Collection |
| `Task3_Part3_SVM.ipynb` | Part 3 | Support Vector Machines | Dry Bean Dataset |

## Workflow (short)

### Part 1 — KNN
- Loaded the wine quality dataset and checked for missing values
- Did a quick EDA on quality distribution and feature correlations
- Encoded wine type and split into train/test
- Compared **No Scaling vs StandardScaler vs MinMaxScaler**
- Tried different **K**, **distance metrics** (Euclidean / Manhattan / Minkowski) and **weights** (uniform / distance)
- Plotted **Accuracy vs K** and **F1 vs K**
- Picked the best combination using F1 (because quality classes are imbalanced)

### Part 2 — Naive Bayes
- Loaded SMS data (tab-separated ham/spam messages)
- Checked class imbalance (more ham than spam)
- Cleaned text: lowercase, removed punctuation, removed stopwords
- Built features with **CountVectorizer** and **TF-IDF**, using **unigrams** and **bigrams**
- Trained MultinomialNB with different **alpha** values: 0.1, 0.5, 1, 2, 5
- Evaluated with Accuracy, Precision, Recall, F1, ROC-AUC and Confusion Matrix

### Part 3 — SVM
- Loaded the dry bean dataset and checked class distribution
- Applied **StandardScaler** before training
- Compared kernels: **Linear, Polynomial, RBF, Sigmoid** and recorded training time
- Tuned **C** and **gamma** for RBF, and **degree** for Polynomial
- Compared models using Accuracy, Precision, Recall, F1 and Confusion Matrix

## How to run
1. Open each notebook in Google Colab / Kaggle / Jupyter
2. Run all cells from top to bottom
3. Datasets are downloaded using `kagglehub` (internet required)

## Main takeaways
- **KNN** depends heavily on scaling and the choice of K
- **Naive Bayes** works well for text after simple cleaning + vectorization; alpha and vectorizer choice change spam recall/precision
- **SVM** needs scaling; RBF usually performs strongly after C/gamma tuning, while sigmoid is often weaker
