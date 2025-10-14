#%%
import os
import logging
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#%%
APPEND_LOG = True  # Set to False to disable log file output
N_SPLITS_LIST = [3, 4, 5, 8, 10]  # Try different n_splits

#%%
# ---------- Logging setup with two handlers ----------
logger = logging.getLogger("my_logger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

log_path = os.path.join(os.path.dirname(__file__), 'run.log')

# Remove any existing handlers to avoid duplicate logs
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
# ---------- End logging setup ----------

#%%
# ---------- NLTK setup ----------
nltk_data_dir = os.path.join(os.path.dirname(__file__))  # Set NLTK data directory to the same directory as this script
nltk.data.path.append(nltk_data_dir)  # Add this directory to NLTK data path
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
# ---------- End NLTK setup ----------

#%%
# Set data directory relative to this script
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data/negative_polarity')
)

#%%
def load_data(data_dir):
    # Load all reviews, labels, and fold numbers
    texts, labels, folds = [], [], []
    for label_dir, label in [('deceptive_from_MTurk', 1), ('truthful_from_Web', 0)]:
        label_path = os.path.join(data_dir, label_dir)  # label_path: data/negative_polarity/deceptive_from_MTurk or truthful_from_Web
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
    # Remove unwanted tokens
    unwanted = {"...", "'s", "``", "'re", "n't", "''", "i.e."}
    tokens = [w for w in tokens if w not in unwanted]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def run_experiment(ngram_range, n_splits_list=[4]):
    texts, labels, folds = load_data(DATA_DIR)
    # test
    # print(f"texts: {texts[:2]}\n labels: {labels[:2]}\n flods: {folds[:2]}")
    texts = [preprocess(t) for t in texts]
    # test
    # print(f"Preprocessed texts: {texts[:2]}")
    labels = np.array(labels)   # Convert to numpy array
    folds = np.array(folds) # Convert to numpy array

    train_idx = np.where(folds < 5)[0]
    test_idx = np.where(folds == 5)[0]
    X_train, y_train = [texts[i] for i in train_idx], labels[train_idx]
    X_test, y_test = [texts[i] for i in test_idx], labels[test_idx]

    param_grid = {
        'vect__max_features': [500, 1000, 2000, 3000, 3500, 4000],    # Limit vocabulary size
        'clf__alpha': [0.01, 0.05, 0.1, 0.5, 1.0]   # Smoothing parameter
    }

    for n_splits in n_splits_list:
        logger.info(f"--- GridSearchCV with StratifiedKFold n_splits={n_splits} ---")
        pipeline = Pipeline([
            ('vect', CountVectorizer(ngram_range=ngram_range)),
            ('clf', MultinomialNB())
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
        log_prob = clf.feature_log_prob_
        top_fake = feature_names[np.argsort(log_prob[1] - log_prob[0])[-5:][::-1]]
        top_genuine = feature_names[np.argsort(log_prob[0] - log_prob[1])[-5:][::-1]]
        logger.info(f"Top 5 fake-indicative words: {top_fake}")
        logger.info(f"Top 5 genuine-indicative words: {top_genuine}")

def main():
    logger.info("========== Unigram only ==========")
    run_experiment((1,1), n_splits_list=N_SPLITS_LIST)
    logger.info("========== Unigram + Bigram ==========")
    run_experiment((1,2), n_splits_list=N_SPLITS_LIST)

#%%
if __name__ == "__main__":
    main()
    logger.info("=== End this run ===\n" + ("\n" * 5))