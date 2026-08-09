# Task 3_Part 2: Naive Bayes (SMS Spam Classification)

1. Objective
Builds a spam-detection classifier for SMS messages using a Multinomial Naive Bayes model. The goal is to understand how different parameters are affecting the model


2. Dataset overview
- Size: 5,573 messages, 2 columns and after cleaning : message (raw text) and spam (binary label).
- No missing values
-Class distribution: 4,826 ham (86%) and 747 spam (13%)  , imbalanced dataset.


3. Pre-Processing
- Cleaning & Label Encoding
- Splited the messages with label ham or spam 
- Checked class distribution with 'value_counts()'.
- Converted the categorical 'label' column into a binary target: 'spam' = 1 and  'ham' = 0.
- Text Preprocessing to lowercase all text , remove punctuation , remove digits, Tokenize
- split data into 80/20 stratified split ('stratify=y') was used to keep the ham/spam ratio in both datasets.

4. Feature Extraction
Two vectorizers were compared :
- CountVectorizer (unigrams, bigrams)
- TfidfVectorizer (unigrams, bigrams)

5. Model Training
Multinomial Naive Bayes classifier was trained for every combination :
- Vectorizer: CountVectorizer, TfidfVectorizer
- N-gram range: unigram (1,1), bigram (2,2)
- Smoothing parameter 'alpha': 0.1, 0.5, 1, 2, 5

which gives 2 × 2 × 5 = 20 total model configurations, each evaluated on Accuracy,precision, Recall, F1-Score, and ROC-AUC on the test set.

6.Hyperparameter Search
A 'GridSearchCV' (cv=5,  F1-score) was run over the same vectorizer/n-gram/alpha grid to independently confirm the best configuration and to produce a confusion matrix for the selected best model.

**Results**

Top configurations (sorted by F1-score, test set)


Vectorizer	         Alpha	     Accuracy	Precision	  Recall	 F1-Score	 ROC-AUC
CountVectorizer()	   0.1	       0.980269	0.926174	 0.926174	 0.926174	 0.973978
TfidfVectorizer()	   0.1	       0.980269	0.963504	 0.885906	 0.923077	 0.986664
CountVectorizer()	   0.5	       0.979372	0.925676	 0.919463	 0.922559	 0.976646
CountVectorizer()	   2.0	       0.978475	0.937063	 0.899329	 0.917808	 0.971873
CountVectorizer()	   1.0	       0.974888	0.911565	 0.899329	 0.905405	 0.975840

Rest all the 20 model test with their metrics in the notebook.



- **Best parameters:** : 'classifier__alpha': 2, 'vectorizer': CountVectorizer(), 'vectorizer__ngram_range': (1, 1)
- **Best cross-validated F1-score:**  : 0.93

A confusion matrix for this best model was plotted on the test set to visualize the
ham/spam misclassifications.

6. Observations & Conclusions

- Low alpha values perform best:   gave the highest F1 and recall.

- CountVectorizer with unigrams and low alpha is the most balanced choice :  gives the best F1-score and the highest recall among the top performers

- TF-IDF tends to trade recall for precision: TF-IDF configuration gives very high precision (often 1.0) but lower recall as alpha grows

- Best-performing pipeline:  CountVectorizer + unigrams + 'alpha = 0.1–0.5'. This combination best balances precision and recall for this imbalanced dataset 

7. Notebook
- 'Task_3_part_2.ipynb' : https://colab.research.google.com/drive/1mPyzqlc-RCHn8U1v11u6mki-ffXfBfyF?usp=sharing

