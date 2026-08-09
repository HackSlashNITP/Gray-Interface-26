# Part 2- SMS Spam Classification using Naive Bayes

## AIM
The objective for this part of the task was to evaluate how different feature extraction techniques (CountVectorizer vs. TF-IDF), n-gram combinations (Unigrams vs. Bigrams), and smoothing parameters (Alpha) affect text classification metrics.

---

## WORKFLOW
1. Exploratory Data Analysis (EDA): 
   - Loaded and cleaned the SMS Spam Collection dataset.
   - Identified class distribution .
   - Encoded target labels: ham to 0 & spam to 1.

2. Text Cleaning:
   - Converted text to lowercase.
   - Removed special characters, digits, and punctuation.
   - Tokenized text and filtered out common NLTK English stopwords.

3. Train-Test Split:
   - Used an 80-20 stratified train-test split to preserve the target class balance in both subsets.

4. Feature Extraction & Hyperparameter Tuning:
   - Created vector representations using CountVectorizer and TfidfVectorizer.
   - Experimented with unigrams (1,1) and unigrams + bigrams (1,2).
   - Tuned Multinomial Naive Bayes Laplace smoothing parameter for alpha in range [0.1, 0.5, 1.0, 2.0, 5.0].

5. Evaluation:
   - Measured Accuracy, Precision, Recall, F1-Score, and ROC-AUC for all combinations.
   - Evaluated model predictions using a Confusion Matrix.

---

## CONCLUSION
- Adding smoothing prevents the zero-probability problem for unseen vocabulary in the test set. Lower alpha values work best when vocabulary is dense.
- High Precision is crucial for spam detection because a False positive is far worse than a False negative.
- TF-IDF down weights common non-stopword tokens which helps in improving precision on text classification tasks.