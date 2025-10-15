#%%
import os
import logging
import string
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

#%%
APPEND_LOG = False

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

def plot_confusion(y_true, y_pred, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['genuine', 'fake'], yticklabels=['genuine', 'fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def run_unigram_only(X_train, y_train, param_grid, vectorizer='count'):
    vect_cls = None
    if vectorizer == 'tfidf':
        vect_cls = TfidfVectorizer
    if vectorizer == 'count':
        vect_cls = CountVectorizer
    vect = vect_cls(ngram_range=(1,1))
    pipeline = Pipeline([
        ('vect', vect),
        ('clf', LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))
    ])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_vect = grid.best_estimator_.named_steps['vect']
    best_vocab = best_vect.vocabulary_
    best_params = grid.best_params_
    logger.info(f"Unigram best params: {best_params}")
    logger.info(f"Unigram best score (mean CV F1): {grid.best_score_:.4f}")
    return best_vocab, best_params

def run_combined_model(X_train, y_train, param_grid, unigram_vocab, vectorizer='count'):
    vect_cls = None
    if vectorizer == 'tfidf':
        vect_cls = TfidfVectorizer
    if vectorizer == 'count':
        vect_cls = CountVectorizer
    unigram_vect = vect_cls(ngram_range=(1,1), vocabulary=unigram_vocab)
    bigram_vect = vect_cls(ngram_range=(2,2))
    combined = FeatureUnion([
        ('unigram', unigram_vect),
        ('bigram', bigram_vect)
    ])
    pipeline = Pipeline([
        ('features', combined),
        ('clf', LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000))
    ])
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipeline, param_grid=param_grid, cv=cv, scoring='f1', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    logger.info(f"Combined best params: {grid.best_params_}")
    logger.info(f"Combined best score (mean CV F1): {grid.best_score_:.4f}")
    return grid.best_estimator_

def run_experiment(vectorizer='count'):
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
    combined_param_grid = {
        'features__bigram__max_features': [500, 1000, 2000, 3000, 3500, 4000],
        'clf__C': [0.001, 0.01, 0.1, 1, 10]
    }

    # Step 1: Unigram only
    logger.info(f"========== Unigram only ({vectorizer}) ==========")
    unigram_vocab, best_unigram_params = run_unigram_only(X_train, y_train, param_grid, vectorizer=vectorizer)

    # Evaluate unigram model on test set
    vect_cls = None
    if vectorizer == 'tfidf':
        vect_cls = TfidfVectorizer
    if vectorizer == 'count':
        vect_cls = CountVectorizer
    best_unigram_vect = vect_cls(ngram_range=(1,1), vocabulary=unigram_vocab)
    unigram_pipeline = Pipeline([
        ('vect', best_unigram_vect),
        ('clf', LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, C=best_unigram_params['clf__C']))
    ])
    unigram_pipeline.fit(X_train, y_train)
    y_pred_uni = unigram_pipeline.predict(X_test)
    logger.info(f"Unigram Test set results (fold 5):")
    logger.info(classification_report(y_test, y_pred_uni, digits=4))
    # Show top features for unigram
    feature_names_uni = best_unigram_vect.get_feature_names_out()
    clf_uni = unigram_pipeline.named_steps['clf']
    coefs_uni = clf_uni.coef_[0]
    top_fake_uni = feature_names_uni[np.argsort(coefs_uni)[-5:][::-1]]
    top_genuine_uni = feature_names_uni[np.argsort(coefs_uni)[:5]]
    logger.info(f"Unigram Top 5 fake-indicative words: {top_fake_uni}")
    logger.info(f"Unigram Top 5 genuine-indicative words: {top_genuine_uni}")

    # Step 2: Combined model (unigram vocab + bigram, FeatureUnion, GridSearchCV)
    logger.info(f"========== Unigram + Bigram (FeatureUnion, {vectorizer}, GridSearchCV) ==========")
    best_combined_model = run_combined_model(X_train, y_train, combined_param_grid, unigram_vocab, vectorizer=vectorizer)

    # Evaluate on test set
    y_pred = best_combined_model.predict(X_test)
    logger.info(f"Test set results (fold 5):")
    logger.info(classification_report(y_test, y_pred, digits=4))

    # Show top features
    unigram_feat = best_combined_model.named_steps['features'].transformer_list[0][1].get_feature_names_out()
    bigram_feat = best_combined_model.named_steps['features'].transformer_list[1][1].get_feature_names_out()
    feature_names = np.concatenate([unigram_feat, bigram_feat])
    clf = best_combined_model.named_steps['clf']
    coefs = clf.coef_[0]
    top_fake = feature_names[np.argsort(coefs)[-5:][::-1]]
    top_genuine = feature_names[np.argsort(coefs)[:5]]
    logger.info(f"Top 5 fake-indicative words: {top_fake}")
    logger.info(f"Top 5 genuine-indicative words: {top_genuine}")

    # --- Visualization: Confusion Matrix ---
    # plot_title = f"Confusion Matrix ({vectorizer}, Unigram+Bigram)"
    # save_path = f"confmat_{vectorizer}_1_2_combined.png"
    # plot_confusion(y_test, y_pred, plot_title, save_path=save_path)
    # --- End Visualization: Confusion Matrix ---

#%%
if __name__ == "__main__":
    run_experiment(vectorizer='count')
    run_experiment(vectorizer='tfidf')
    logger.info("=== End this run ===\n" + ("\n" * 5))