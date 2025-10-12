
import glob as glob
import joblib, os
from tempfile import mkdtemp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.stats import randint

def train_models(x_train, y_train, fold_train, save_dir="models"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    uni_pipe = Pipeline([
        ("vectorizer", TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1,1))),
        ("classifier", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ])
    uni_pipe.memory = mkdtemp()

    uni_grid = {
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": randint(10,51),
        "classifier__min_samples_leaf": randint(2,11),
        "classifier__ccp_alpha":[0.0, 1e-4, 3e-4, 1e-3, 3e-3],

        "vectorizer__min_df": [1, 2, 5],
        "vectorizer__max_features": [None, 10000, 20000],
        "vectorizer__max_df": [1.0, 0.95, 0.9]
    }
    cv = GroupKFold(n_splits=4)

    uni_model = RandomizedSearchCV(
        uni_pipe, uni_grid, cv = cv, n_iter =30, scoring = "f1", n_jobs=1, random_state=42, verbose=1
    )
    print("Training unigram model....")
    uni_model.fit(x_train, y_train, groups=fold_train)

    #unigram+bigram
    uni_vec = uni_model.best_estimator_.named_steps["vectorizer"]
    uni_vocab = uni_vec.vocabulary_
    #Bigram
    bi_pipe = Pipeline([
        ("features", FeatureUnion([
            ("uni_vec", TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1,1), vocabulary=uni_vocab)),
            ("bi_vec", CountVectorizer(stop_words="english", lowercase=True, ngram_range=(2,2), binary=True))])),
        ("classifier", DecisionTreeClassifier(class_weight="balanced", random_state=42))
    ])
    bi_pipe.memory = mkdtemp()

    bi_grid = {
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__max_depth": [None, 5, 10, 20],
        "classifier__min_samples_split": randint(10,51),
        "classifier__min_samples_leaf": randint(2,11),
        "classifier__ccp_alpha":[0.0, 1e-4, 3e-4, 1e-3, 3e-3],

        "features__bi_vec__min_df": [2, 5],
        "features__bi_vec__max_features": [None, 10000, 20000],
        "features__bi_vec__max_df": [1.0, 0.95]
    }

    bi_model = RandomizedSearchCV(
        bi_pipe, bi_grid, cv = cv, n_iter =30, scoring = "f1", n_jobs=1, random_state=42, verbose=1
    )
    print("Training bigram model....")
    bi_model.fit(x_train, y_train, groups=fold_train)

    joblib.dump(uni_model, os.path.join(save_dir, "unigram.joblib"))
    joblib.dump(bi_model, os.path.join(save_dir, "bigram.joblib"))

    print(f"Saved models to {save_dir}")
    return uni_model, bi_model

if __name__ == "__main__":
    print("Training models...")