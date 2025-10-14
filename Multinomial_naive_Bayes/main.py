#%%
import os
import logging
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#%%
APPEND_LOG = True  # Set to False to disable log file output

# N_SPLITS_LIST = [3, 4, 5, 8, 10]  # Using different n_splits is not improving performance in this case, so we keep it simple
N_SPLITS_LIST = [4]

# Set data directory relative to this script
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../data/negative_polarity')
)

#%% ---------- Logging setup with two handlers ----------
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

#%% ---------- NLTK setup ----------
nltk_data_dir = os.path.join(os.path.dirname(__file__))  # Set NLTK data directory to the same directory as this script
nltk.data.path.append(nltk_data_dir)  # Add this directory to NLTK data path
nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('punkt_tab', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)
# ---------- End NLTK setup ----------

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
    text_lower = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    unwanted = {"...", "'s", "``", "'re", "n't", "''", "i.e."}      # Remove unwanted tokens
    tokens = nltk.word_tokenize(text_lower)
    tokens = [w for w in tokens if w not in string.punctuation]
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [w for w in tokens if w not in unwanted]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def run_unigram_only(X_train, y_train, param_grid, vectorizer_type='count'):
    if vectorizer_type == 'count':
        vect = CountVectorizer(ngram_range=(1,1))
    else:
        vect = TfidfVectorizer(ngram_range=(1,1))
    pipeline = Pipeline([
        ('vect', vect),
        ('clf', MultinomialNB())
    ])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_vect = grid.best_estimator_.named_steps['vect']
    best_vocab = best_vect.vocabulary_
    logger.info(f"Unigram best params: {grid.best_params_}")
    logger.info(f"Unigram best score (mean CV F1): {grid.best_score_:.4f}")
    return best_vocab, grid.best_params_

def run_combined_model(X_train, y_train, param_grid, unigram_vocab, vectorizer_type='count'):
    if vectorizer_type == 'count':
        unigram_vect = CountVectorizer(ngram_range=(1,1), vocabulary=unigram_vocab)
        bigram_vect = CountVectorizer(ngram_range=(2,2))
    else:
        unigram_vect = TfidfVectorizer(ngram_range=(1,1), vocabulary=unigram_vocab)
        bigram_vect = TfidfVectorizer(ngram_range=(2,2))
    combined = FeatureUnion([
        ('unigram', unigram_vect),
        ('bigram', bigram_vect)
    ])
    pipeline = Pipeline([
        ('features', combined),
        ('clf', MultinomialNB())
    ])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    logger.info(f"Combined best params: {grid.best_params_}")
    logger.info(f"Combined best score (mean CV F1): {grid.best_score_:.4f}")
    return grid.best_estimator_

def run_experiment():
    texts, labels, folds = load_data(DATA_DIR)
    # print(f"texts: {texts[:2]}\n labels: {labels[:2]}\n flods: {folds[:2]}")  # test
    texts = [preprocess(t) for t in texts]
    # print(f"Preprocessed texts: {texts[:2]}") # test
    labels = np.array(labels)   # Convert to numpy array
    folds = np.array(folds) # Convert to numpy array

    train_idx = np.where(folds < 5)[0]
    test_idx = np.where(folds == 5)[0]
    X_train, y_train = [texts[i] for i in train_idx], labels[train_idx]
    X_test, y_test = [texts[i] for i in test_idx], labels[test_idx]

    # Step 1: Unigram only
    logger.info("========== Unigram only (CountVectorizer) ==========")
    unigram_param_grid = {
        'vect__max_features': [500, 1000, 2000, 3000, 3500, 4000],
        'clf__alpha': [0.01, 0.05, 0.1, 0.5, 1.0]
    }
    unigram_vocab, best_unigram_params = run_unigram_only(X_train, y_train, unigram_param_grid, vectorizer_type='count')

    # Evaluate unigram model on test set
    best_unigram_vect = CountVectorizer(ngram_range=(1,1), vocabulary=unigram_vocab)
    unigram_pipeline = Pipeline([
        ('vect', best_unigram_vect),
        ('clf', MultinomialNB(alpha=best_unigram_params['clf__alpha']))
    ])
    unigram_pipeline.fit(X_train, y_train)
    y_pred_uni = unigram_pipeline.predict(X_test)
    logger.info(f"Unigram Test set results (fold 5):")
    logger.info(classification_report(y_test, y_pred_uni, digits=4))
    # Show top features for unigram
    feature_names_uni = best_unigram_vect.get_feature_names_out()
    clf_uni = unigram_pipeline.named_steps['clf']
    log_prob_uni = clf_uni.feature_log_prob_
    top_fake_uni = feature_names_uni[np.argsort(log_prob_uni[1] - log_prob_uni[0])[-5:][::-1]]
    top_genuine_uni = feature_names_uni[np.argsort(log_prob_uni[0] - log_prob_uni[1])[-5:][::-1]]
    logger.info(f"Unigram Top 5 fake-indicative words: {top_fake_uni}")
    logger.info(f"Unigram Top 5 genuine-indicative words: {top_genuine_uni}")

    # Step 2 & 3: Combined model (unigram vocab + bigram, FeatureUnion, GridSearchCV)
    logger.info("========== Unigram + Bigram (FeatureUnion, GridSearchCV) ==========")
    param_grid = {
        'features__bigram__max_features': [500, 1000, 2000, 3000, 3500, 4000],
        'clf__alpha': [0.01, 0.05, 0.1, 0.5, 1.0]
    }
    best_combined_model = run_combined_model(X_train, y_train, param_grid, unigram_vocab, vectorizer_type='count')

    # Evaluate on test set
    y_pred = best_combined_model.predict(X_test)
    logger.info(f"Test set results (fold 5):")
    logger.info(classification_report(y_test, y_pred, digits=4))

    # Show top features
    unigram_feat = best_combined_model.named_steps['features'].transformer_list[0][1].get_feature_names_out()
    bigram_feat = best_combined_model.named_steps['features'].transformer_list[1][1].get_feature_names_out()
    feature_names = np.concatenate([unigram_feat, bigram_feat])
    clf = best_combined_model.named_steps['clf']
    log_prob = clf.feature_log_prob_
    top_fake = feature_names[np.argsort(log_prob[1] - log_prob[0])[-5:][::-1]]
    top_genuine = feature_names[np.argsort(log_prob[0] - log_prob[1])[-5:][::-1]]
    logger.info(f"Top 5 fake-indicative words: {top_fake}")
    logger.info(f"Top 5 genuine-indicative words: {top_genuine}")

    
#%%
if __name__ == "__main__":
    run_experiment()
    logger.info("=== End this run ===\n" + ("\n" * 5))

# %%
