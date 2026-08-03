# Task 3: Classification Algorithms

Three notebooks exploring KNN, Naive Bayes, and SVM — instance-based, probabilistic, and margin-based learning respectively.

---

## Part 1: K-Nearest Neighbors — Wine Quality

**Google Colab Part 1:** https://colab.research.google.com/drive/1pOH10wMJ23_K-jkr8nkZrfJWrOZVwvdC?usp=sharing

**Dataset:** UCI Red & White Wine Quality (6,497 rows, 13 columns — 11 physicochemical features, `type` red/white, target `quality` score 3–9). No missing values.

**Key EDA/Preprocessing steps:**
- Combined red and white wine CSVs into one frame with a `type` column.
- Checked `describe()`/`info()`/`isnull()` — clean numeric data, no nulls.
- One-hot encoded `type`; compared No Scaling vs StandardScaler vs MinMaxScaler on numeric features.
- Ran K from 1–20 with 5-fold cross-validation, and grid-searched over `n_neighbors`, `weights` (uniform/distance), and `metric` (euclidean/manhattan/minkowski).

**Best configuration & conclusion:**
Unscaled features performed poorly (Accuracy ≈ 0.457) since KNN's distance metric was dominated by large-range features like `total sulfur dioxide`. StandardScaler gave the best result (Accuracy ≈ 0.671, F1 ≈ 0.659), narrowly ahead of MinMaxScaler (≈ 0.668). GridSearchCV confirmed the optimal hyperparameters as **K=20, metric='manhattan', weights='distance'**. Accuracy improved steadily as K increased, plateauing around K=18–20 — larger K smooths out noise in this fairly overlapping, multi-class quality target, while `weights='distance'` lets closer neighbors matter more, and Manhattan distance handles the mixed-scale features better than Euclidean.

---

## Part 2: Naive Bayes — SMS Spam Classification

**Google Colab Part 2:** https://colab.research.google.com/drive/1AuHKa3Keg0Weai5HDu8g9sGx_BHAkEzu?usp=sharing

**Dataset:** SMS Spam Collection (5,572 messages, labeled `ham`/`spam`). No missing values; 403 duplicate rows removed, leaving 4,516 ham vs 653 spam (imbalanced, ~87/13 split).

**Key EDA/Preprocessing steps:**
- Checked class distribution (pie chart) — confirmed strong class imbalance toward `ham`.
- Added `message_length` and `word_count` features for exploration.
- Text cleaning: lowercased, stripped punctuation and digits, removed stopwords, tokenized.
- Built 4 feature sets: CountVectorizer/TF-IDF × Unigram/Bigram.
- Trained MultinomialNB across `alpha` ∈ {0.1, 0.5, 1, 2, 5} for every vectorizer, evaluated with Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.

**Best pipeline & conclusion:**
The best-performing setup was **TF-IDF with Bigrams, alpha=0.1** (Accuracy ≈ 0.985, Precision ≈ 0.991, Recall ≈ 0.890, F1 ≈ 0.938, ROC-AUC ≈ 0.982). TF-IDF outperformed raw CountVectorizer because it down-weights common words and highlights terms that are more distinctive of spam (e.g., "free", "win", "claim"), while bigrams capture short spam phrases (like "free entry") that unigrams miss. Lower alpha values consistently won: with the vocabulary already large and spam being the minority class, less Laplace smoothing preserves the sharp probability differences between spam/ham words instead of flattening them toward the prior — higher alpha values overly smoothed the model, causing recall to collapse (e.g., alpha=5 with TF-IDF-Bigram dropped recall to ~0.047 despite perfect precision).

---

## Part 3: Support Vector Machines — Dry Bean Classification

**Google Colab Part 3:** https://colab.research.google.com/drive/1zmhYf54Fc9fOyzuv-iTOGjNuYCdtMc9r?usp=sharing

**Dataset:** Dry Bean Dataset (13,611 rows, 16 numeric shape/geometric features, target `Class` — 7 bean varieties: DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY). No missing values; moderately imbalanced classes (BOMBAY has only 522 samples vs 3,546 for DERMASON).

**Key EDA/Preprocessing steps:**
- Verified no nulls and inspected feature ranges via `describe()` — features vary widely in scale (Area in the tens of thousands vs Eccentricity between 0–1).
- Class distribution checked via `value_counts()`.
- Applied StandardScaler to all numeric features in a pipeline before training.
- Trained SVMs with Linear, Polynomial, RBF, and Sigmoid kernels, grid-searching `C`, `gamma`, and (for polynomial) `degree`, and timing each kernel's training.

**Best kernel & conclusion:**

| Kernel | Best Params | Train Time | Accuracy | F1 (weighted) |
|---|---|---|---|---|
| Linear | C=10 | ~12s | 0.9291 | 0.9294 |
| Polynomial | C=10, degree=2, gamma=0.1 | ~4m 12s | 0.9295 | 0.9298 |
| **RBF** | **C=10, gamma=0.1** | **~49s** | **0.9328** | **0.9330** |
| Sigmoid | C=1, gamma=0.01 | ~1m 14s | 0.9192 | 0.9196 |

The **RBF kernel** performed best overall, edging out Linear and Polynomial on accuracy/F1 while training far faster than Polynomial. This makes sense because the bean classes are separable but not perfectly linearly separable — RBF's ability to form flexible, localized non-linear boundaries lets it resolve overlap between visually similar bean types (e.g., SIRA vs DERMASON), without the extra computational cost of higher-degree polynomial kernels. Sigmoid performed worst, as it is less naturally suited to multi-class geometric feature spaces like this one. Most confusion occurred between shape-similar classes (e.g., DERMASON/SIRA/HOROZ), consistent with their overlapping physical geometry.
