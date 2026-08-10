# SMS Spam Detection: Multinomial Naive Bayes Comparison

This project implements an end-to-end NLP pipeline to classify SMS messages as 'spam' or 'ham' using the SMS Spam Collection dataset.

### Project Overview
- **Objective**: Identify the optimal preprocessing and vectorization strategy for spam detection.
- **Dataset**: 5,572 messages (~86.6% ham, 13.4% spam).
- **Algorithm**: Multinomial Naive Bayes (MNB).

### Methodology
1.  **Preprocessing**: Lowercasing, punctuation removal, and stopword filtering via NLTK.
2.  **Feature Extraction**: Comparison of four configurations:
    - `CountVectorizer` (Unigrams)
    - `CountVectorizer` (Bigrams)
    - `TfidfVectorizer` (Unigrams)
    - `TfidfVectorizer` (Bigrams)
3.  **Hyperparameter Tuning**: Grid search over Laplace smoothing parameter `alpha` (0.1, 0.5, 1, 2, 5).
4.  **Evaluation**: Metrics include Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### Key Findings
- **Top Model**: `CountVectorizer` (Unigrams) with `alpha=0.1` achieved the highest F1-Score (~0.941).
- **Insights**:
    - **Count vs TF-IDF**: For short SMS text, raw frequencies (CountVectorizer) often outperform normalized weights (TF-IDF).
    - **Smoothing**: Low alpha values are preferred, as they allow the model to better distinguish rare but highly indicative spam keywords.
    - **N-Grams**: Unigrams provide a robust baseline; while Bigrams increase complexity and ROC-AUC, they showed marginal gains in F1-Score for this specific dataset.

### Results Summary
| Vectorization | Alpha | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CV Unigrams** | **0.1** | **98.48%** | **97.14%** | **91.28%** | **0.941** |