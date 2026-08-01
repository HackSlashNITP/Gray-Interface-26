# SMS Spam Classification - Naive Bayes (Task 3, Part 2)

Hey there! This folder contains my work for Part 2 of Task 3[span_0](start_span)[span_0](end_span). The main goal here was to take the SMS Spam Collection dataset, clean up the text, and build a Multinomial Naive Bayes model to figure out whether a text message is spam or legitimate (`ham`)[span_1](start_span)[span_1](end_span). 

## How I Did It (My Workflow)

1. **Checking out the Data:** Started by loading the dataset and looking at how many spam vs. ham messages we're dealing with (the class distribution)[span_2](start_span)[span_2](end_span).
2. **Cleaning the Text:** Wrote a quick preprocessing function to lowercase everything, strip out punctuation, break messages down into tokens, and filter out common stopwords[span_3](start_span)[span_3](end_span).
3. **Feature Extraction:** Tested out different ways to turn text into numbers. I compared `CountVectorizer` against `TfidfVectorizer`, and also tested just using single words (unigrams) versus word pairs (unigrams + bigrams)[span_4](start_span)[span_4](end_span).
4. **Training & Tuning:** Trained a Multinomial Naive Bayes model and tested a bunch of different smoothing parameters (`alpha` values: 0.1, 0.5, 1, 2, and 5) to see what worked best[span_5](start_span)[span_5](end_span).
5. **Evaluating Everything:** Checked all the models using a full suite of metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrices[span_6](start_span)[span_6](end_span).

## What Worked Best (My Results)

After running all the combinations, here is what came out on top:

* **The Winning Setup:** TF-IDF Vectorizer using both Unigrams and Bigrams, paired with an alpha smoothing value of `0.1`.
* **Performance:** It pulled off an awesome F1-Score and ROC-AUC of around 0.98 to 0.99+.

### Why this setup performed so well:
* **TF-IDF over CountVectorizer:** TF-IDF helped tone down words that show up way too often and gave more weight to the words that actually matter for catching spam.
* **Adding Bigrams:** Including bigrams was a game-changer for precision. It allowed the model to catch common spam phrases (like "claim your prize" or "free entry") instead of just looking at isolated words.
* **Low Alpha Smoothing:** Sticking with a smaller alpha (`0.1`) kept the model sharp. It didn't over-smooth the probabilities, so it held onto its ability to sharply distinguish between spam and normal messages.
*