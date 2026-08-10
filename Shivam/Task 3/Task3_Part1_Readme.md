# Part 1: K-Nearest Neighbors — Wine Quality Classification

## Dataset

[Wine Quality (Red and White)](https://www.kaggle.com/datasets/sinadehestani/wine-quality-red-and-white) — 6,497 samples, 11 physicochemical features (acidity, sugar, sulfur dioxide, density, pH, alcohol, etc.), a `type` column (red/white), and a `quality` score from 3–9. No missing values.

## Problem Framing

The original `quality` column is a 7-class target, but the class distribution is heavily skewed — quality 9 has only 5 samples, quality 3 has 30, while 5/6/7 together account for over 90% of the data. Training KNN directly on this would mostly measure how well the model handles class imbalance rather than how well the hyperparameters actually work, since there's barely enough data for KNN to learn what separates a 9 from a 7.

To get a cleaner signal, I converted this into binary classification: `quality >= 7` is labeled "good" (1), everything else is "not good" (0). This gives a 1,277 vs 5,220 split — still imbalanced, but workable, and it keeps the focus on what the assignment is actually testing: scaling, K, distance metric, and weighting.

## Preprocessing

- Encoded `type` (red/white) as a binary numeric column.
- Dropped the original `quality` column from the feature set to avoid leaking the label.
- Split the data 80/20 with stratification on the target, so both train and test sets keep the same ~80/20 class ratio.
- Built three versions of the feature set: no scaling, StandardScaler, and MinMaxScaler, fitting each scaler only on the training data before transforming both sets.

## EDA Highlights

Alcohol content showed the strongest correlation with quality (0.39) and the clearest separation in boxplots — good wines cluster around a higher median alcohol level. Density and volatile acidity were next, both negatively correlated with quality. Sulphates, citric acid, and pH showed almost no separation between classes and did little discriminative work on their own. Free and total sulfur dioxide were strongly correlated with each other (0.72), which is worth noting as multicollinearity, though it doesn't directly hurt KNN the way it would a linear model.

## Model Experiments

Trained KNN across 162 combinations: 3 scaling methods × 9 K values (1–31) × 3 distance metrics (Euclidean, Manhattan, Minkowski with p=3 to keep it distinct from Euclidean) × 2 weighting schemes (uniform, distance).

**Weighting was the single biggest factor.** Uniform-weighted models degraded sharply as K grew — F1 dropped to around 0.1–0.2 by K=31 — because at high K, distant and less relevant neighbors get an equal vote, and the majority class drowns out the minority. Distance weighting stayed stable across the whole K range since closer neighbors get more say.

**Scaling mattered, but less dramatically.** No-scaling models performed reasonably at low K but fell apart at high K when combined with uniform weighting. StandardScaler and MinMaxScaler both kept F1 relatively flat and high across K, and performed almost identically to each other — this dataset doesn't have feature ranges extreme enough for the choice between the two to matter much.

**Distance metric had the smallest impact.** Euclidean, Manhattan, and Minkowski tracked closely together in nearly every configuration.

## Best Configuration

**StandardScaler + K=7 + Manhattan distance + distance weighting** — F1 = 0.665, Accuracy = 0.878. A K=31/Euclidean combination edged it slightly on raw accuracy (0.885), but with a lower F1, so it was skipped in favor of the more balanced model, since accuracy alone is a misleading metric on an 80/20 imbalanced target.

Classification report for the final model:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Not Good | 0.91 | 0.94 | 0.93 |
| Good | 0.72 | 0.62 | 0.67 |

The model catches 62% of actual "good" wines with 72% precision on the ones it flags. Reasonable given the imbalance, but there's clear room to improve — techniques like SMOTE or a class-weighted variant would be a natural next step.

## Files

- `knn_wine_quality.ipynb` — full notebook (EDA, preprocessing, experiments, plots, final model)