# Gray Interface '26 — Task 3: KNN, Naive Bayes & SVM

---

## Part 1 · K-Nearest Neighbors (KNN)

**Dataset:** Wine Quality (Red + White combined)

### Preprocessing
- Combined red and white wine CSVs, added a `wine_type` column encoded as 0/1.
- Converted `quality` score into binary classification: **good (≥7) vs not good (<7)**. This makes the problem cleaner — predicting a single binary label is more practical than predicting 10 quality grades.
- 80/20 stratified train-test split.

### Effect of Scaling

| Scaler | Why it matters |
|---|---|
| No Scaling | Features with large ranges (e.g. `total sulfur dioxide`: 0–300) dominate distance — results are poor |
| StandardScaler | Centers and scales each feature — best overall for KNN |
| MinMaxScaler | Squeezes all features to [0,1] — works well but slightly less robust to outliers |

KNN is purely distance-based, so scaling is not optional — it's essential.

### Distance Metrics
Euclidean and Minkowski (p=2) are mathematically identical and both perform well. Manhattan (p=1) is slightly more robust to outliers but gives similar results here.

### Weighting
`weights='distance'` consistently outperforms `weights='uniform'` — nearer neighbors are more reliable classifiers and should have more say in the vote.

### Best Configuration
- **Scaler:** StandardScaler
- **Metric:** Euclidean
- **Weights:** Distance
- **Best K:** Found from F1 vs K plot (typically 7–15 range)

Low K overfits (noisy boundary), high K underfits (too smooth). The F1 vs K plot shows the optimal K clearly.

---

## Part 2 · Naive Bayes (SMS Spam Detection)

**Dataset:** SMS Spam Collection

### Text Preprocessing Steps
1. Lowercased all text
2. Removed punctuation and numbers
3. Removed English stopwords (words like "the", "is" that carry no spam signal)
4. Tokenized by splitting on whitespace

### Vectorizers

| Vectorizer | How it works |
|---|---|
| CountVectorizer | Raw word counts — treats all words equally |
| TF-IDF | Weights words by how unique they are to a document — down-weights common words |

TF-IDF outperforms CountVectorizer here because spam has distinctive vocabulary ("free", "win", "prize") that TF-IDF rewards by giving them higher weights.

### N-grams
- **Unigrams**: single words — catch individual spam keywords
- **Bigrams**: word pairs — catch phrases like "free entry", "prize winner" that are strong spam signals

Bigrams generally improved Precision (fewer false positives).

### Smoothing (Alpha)
Alpha prevents zero probability for unseen words. Too low (0.1) → overfit to training vocab. Too high (5.0) → all words treated almost equally, loses discriminative power. Alpha=0.5–1.0 is the sweet spot.

### Key Observation
The dataset is ~87% ham / 13% spam. Accuracy alone is misleading — a model saying "ham" for everything gets 87% accuracy but is useless. **F1 and Precision/Recall** are the meaningful metrics.

---

## Part 3 · Support Vector Machines (SVM)

**Dataset:** Dry Bean Dataset (7 bean classes, 16 shape features)

### Preprocessing
- Loaded from Excel file using `openpyxl`.
- Label encoded the 7 bean classes.
- StandardScaler applied — SVM kernel distances are meaningless without scaling.
- 80/20 stratified split.

### Kernel Comparison

| Kernel | Strength | Weakness |
|---|---|---|
| Linear | Fast, interpretable | Fails when classes aren't linearly separable |
| RBF | Best general-purpose kernel, handles non-linear boundaries | Needs tuning of C and gamma |
| Polynomial | Captures curved boundaries | Slow at high degrees, can overfit |
| Sigmoid | Sometimes works like a neural network layer | Often unstable for multi-class |

**RBF won** because the 7 bean classes have overlapping physical measurements — a non-linear boundary is needed to separate them cleanly.

### C Parameter
- Low C: Wide margin, more misclassifications allowed — underfits.
- High C: Narrow margin, fits training data tightly — risk of overfitting.
- Best: C=10 with RBF on this dataset.

### Gamma Parameter (RBF)
- High gamma: Each point influences only a tiny neighbourhood — overfits.
- Low gamma: Very smooth boundary — underfits.
- `gamma='scale'` (auto-calculated from data variance) was the most reliable choice.

### Polynomial Degree
Higher degrees capture more complex boundaries but take significantly longer to train. Degree=3 gave the best F1/time tradeoff.

### Why Feature Scaling is Critical for SVM
SVM computes dot products in kernel space — features with large absolute values dominate the kernel computation. Without StandardScaler, `Area` would completely overwhelm `Compactness`, making the kernel meaningless.

---

## Repo Structure

```
Kshitiz/
├── Task3_Part1_KNN.ipynb
├── Task3_Part2_NaiveBayes.ipynb
├── Task3_Part3_SVM.ipynb
└── README_Task3.md
```
