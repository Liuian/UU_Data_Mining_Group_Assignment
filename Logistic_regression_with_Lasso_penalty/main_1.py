"""
This script implements and evaluates a Logistic Regression classifier with an L1 (Lasso) penalty
for deceptive opinion spam detection. It follows a multi-stage experimental design
to compare the performance of different feature sets (unigrams vs. unigrams+bigrams)
and vectorization methods (CountVectorizer vs. TfidfVectorizer).

Experimental Design:

The `main` function orchestrates four independent experiments by calling the
`run_experiment` function with different parameters:

1.  Unigram Only (`ngram_range=(1,1)`):
    - A vectorizer (`CountVectorizer` or `TfidfVectorizer`) is configured with `ngram_range=(1,1)`.
    - It builds a candidate vocabulary consisting solely of unigrams (single words)
      from the training data.
    - `GridSearchCV` then searches for the best `max_features` by selecting the
      top N most frequent/important features from this candidate pool.

2.  Unigram + Bigram (`ngram_range=(1,2)`):
    - A new vectorizer is configured with `ngram_range=(1,2)`.
    - It builds a new, mixed candidate vocabulary containing *both* unigrams
      and bigrams from the training data.
    - `GridSearchCV` again searches for the best `max_features`. However, in this
      run, it selects the top N features from the *mixed* pool, where unigrams
      and bigrams compete for inclusion based on their overall frequency/importance.

Key Distinction:
The core difference lies in how the final features are selected:

-   **Identical Candidate Pool for Unigrams:** Before hyperparameter tuning,
    the pool of *all possible* unigram features is identical for both
    the unigram-only and the unigram+bigram experiments because they both
    process the same training data.

-   **Different Competitive Environments:**
    - In the unigram-only experiment, unigrams only compete against other unigrams
      for a spot in the final `max_features` set.
    - In the unigram+bigram experiment, unigrams must compete against bigrams for
      those same spots.

As a result, the final set of unigram features chosen for the unigram+bigram model is
not guaranteed to be the same as the set chosen for the unigram-only model. This
design differs from a `FeatureUnion` approach, where a pre-selected unigram
vocabulary is explicitly combined with a separately generated bigram vocabulary.
"""
#%%
import os
import logging
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#%%
APPEND_LOG = True
N_SPLITS_LIST = [3, 4, 5, 8, 10]

#%% Logging setup (same as multinomial NB)
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_path = os.path.join(os.path.dirname(__file__), 'run.log')
if logger.hasHandlers():
    logger.handlers.clear()
if APPEND_LOG:
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#%% NLTK setup
nltk_data_dir = os.path.join(os.path.dirname(__file__))
nltk.data.path.append(nltk_data_dir)
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

#%%
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data/negative_polarity')
)

#%%
def load_data(data_dir):
    texts, labels, folds = [], [], []
    for label_dir, label in [('deceptive_from_MTurk', 1), ('truthful_from_Web', 0)]:
        label_path = os.path.join(data_dir, label_dir)
        for fold_name in sorted(os.listdir(label_path)):
            fold_num = int(fold_name.replace('fold', ''))
            fold_path = os.path.join(label_path, fold_name)
            for fname in sorted(os.listdir(fold_path)):
                fpath = os.path.join(fold_path, fname)
                with open(fpath, encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(label)
                folds.append(fold_num)
    return texts, labels, folds

def preprocess(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w not in stop_words]
    unwanted = {"...", "'s", "``", "'re", "n't", "''", "i.e."}
    tokens = [w for w in tokens if w not in unwanted]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def run_experiment(ngram_range, n_splits_list=[4], use_tfidf=False):
    texts, labels, folds = load_data(DATA_DIR)
    texts = [preprocess(t) for t in texts]
    labels = np.array(labels)
    folds = np.array(folds)

    train_idx = np.where(folds < 5)[0]
    test_idx = np.where(folds == 5)[0]
    X_train, y_train = [texts[i] for i in train_idx], labels[train_idx]
    X_test, y_test = [texts[i] for i in test_idx], labels[test_idx]

    param_grid = {
        'vect__max_features': [500, 1000, 2000, 3000, 3500, 4000],
        'clf__C': [0.001, 0.01, 0.1, 1, 10]
    }
    vect_cls = TfidfVectorizer if use_tfidf else CountVectorizer

    for n_splits in n_splits_list:
        logger.info(f"--- GridSearchCV with StratifiedKFold n_splits={n_splits} ---")
        pipeline = Pipeline([
            ('vect', vect_cls(ngram_range=ngram_range)),
            ('clf', LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))
        ])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
        grid.fit(X_train, y_train)

        logger.info(f"Best params: {grid.best_params_}")
        logger.info(f"Best score (mean CV F1): {grid.best_score_:.4f}")

        best_model = grid.best_estimator_
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        logger.info(f"Test set results (fold 5):")
        logger.info(classification_report(y_test, y_pred, digits=4))

        vect = best_model.named_steps['vect']
        clf = best_model.named_steps['clf']
        feature_names = np.array(vect.get_feature_names_out())
        coefs = clf.coef_[0]
        top_fake = feature_names[np.argsort(coefs)[-5:][::-1]]
        top_genuine = feature_names[np.argsort(coefs)[:5]]
        logger.info(f"Top 5 fake-indicative words: {top_fake}")
        logger.info(f"Top 5 genuine-indicative words: {top_genuine}")

def main():
    logger.info("========== Unigram only (CountVectorizer) ==========")
    run_experiment((1,1), n_splits_list=N_SPLITS_LIST, use_tfidf=False)
    logger.info("========== Unigram + Bigram (CountVectorizer) ==========")
    run_experiment((1,2), n_splits_list=N_SPLITS_LIST, use_tfidf=False)
    logger.info("========== Unigram only (TFIDF) ==========")
    run_experiment((1,1), n_splits_list=N_SPLITS_LIST, use_tfidf=True)
    logger.info("========== Unigram + Bigram (TFIDF) ==========")
    run_experiment((1,2), n_splits_list=N_SPLITS_LIST, use_tfidf=True)

#%%
if __name__ == "__main__":
    main()
    logger.info("======================= End this run ==========================\n" + ("\n" * 5))
