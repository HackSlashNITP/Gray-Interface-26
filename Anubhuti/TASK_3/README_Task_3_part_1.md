# Task 3_Part 1: K-Nearest Neighbors (KNN)

Dataset: Wine Quality (Red and White) 

1. Objective
Predict wine 'quality' (multi-class: scores 3–9)  given features and see how changing hyperparameter affects the KNN's performance.

2. Dataset Overview
-'shape' : 6,497 rows × 13 columns 
-'type': red (1,599) and white (4,898)
-'quality': 7 classes, heavily imbalanced only 5 and 6 make up almost 77% of samples, while quality 3 and 9 are extremely low.
- No missing values found.
-1,177 duplicate rows were identified and removed (6,497 → 5,320 rows).

3. Preprocessing
-Checked and confirmed no missing values.
-Identified and removed duplicate rows before splitting.
-Label-encoded 'type' (red=0, white=1).
-Checked correlation of each feature with 'quality' - top correlates were 'alcohol: (0.44)','density:(0.31)'.
-Split data 80/20 with stratification on 'quality' to preserve the class imbalance ratio in both train and test datasets.

4. Feature Scaling Comparison
Three conditions were compared using 'GridSearchCV' (cv= 5) over K, distance metric, and weighting:

without scaling : Best Parameters {'metric': 'minkowski', 'n_neighbors': 1, 'p': 3, 'weights': 'uniform'} best f1_macro score :  0.2037327040147227
MinMax Scaling : Best Parameters {'metric': 'minkowski', 'n_neighbors': 6, 'p': 3, 'weights': 'distance'} best f1_macro  score :  0.26567936424696237
Standard Scaling : Best Parameters {'metric': 'minkowski', 'n_neighbors': 6, 'p': 3, 'weights': 'distance'} best f1_macro  score :  0.2688764289644387



Observation: Unlike accuracy, F1-macro didn't improve much with scaling and it's may be  because of  f1-macro is being dragged down by rare quality classes (3, 4, 8, 9), and also scaling doesn't fix the core problem that for specific classes we don't have much info,even though the data were scaled . Scaling helps KNN measure distance more fairly between features, but it can't create more examples of quality 9 wines.

5. Hyperparameter Comparison
- K (n_neighbors): tested in range of 1-99 in steps of 5.
- Distance metric: Euclidean, Manhattan, Minkowski.
- Weighting: Uniform, Distance.
- Across all three scaling conditions, Manhattan distance consistently outperformed the alternatives.

Accuracy vs K and F1-Score (macro) vs K were plotted side by side for all three scaling
conditions.

observation : In Accuracy Vs K graph with increase in k value the Accuracy also increases in start and become so inconsistent whether it is scaled or not , without scaled data have low accuracy while the scaled one have high value of accuracy as compared to non-scaled ones , for MinMax scaling the accuracy value is grate mainly. 
While in F1_score(macro) Vs K graph with increase in k value , f1_score decreases and the heighest value is for the MinMax scaling and for all the scaled or non-scaled the value decrese mostly but somewhere it get increse also locally , but the maximum values for each scaling is for the low value of Ks

6. Why Macro F1, not just Accuracy
Because quality is heavily imbalanced (very few samples at 3, 4, 8, 9), accuracy can be misleading - a model could score well by favoring the dominant classes (5, 6) while ignoring rare ones. F1-macro (which weights every class equally regardless of size) was tracked alongside accuracy.


Link to the notebook : https://colab.research.google.com/drive/1AktqyPQVujQV_4S39dWUTd1bYG9LuzZN?usp=sharing
