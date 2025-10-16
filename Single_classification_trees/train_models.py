
import glob as glob
import joblib, os
from tempfile import mkdtemp

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from scipy.stats import randint
from sklearn.base import clone

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
    best_uni_vec = uni_model.best_estimator_.named_steps["vectorizer"]
    uni_vec_for_union = clone(best_uni_vec)

    
    bi_vec_for_union = TfidfVectorizer(
    lowercase=True,
    ngram_range=(2, 2),
    stop_words=None,          # keep "not", "no", etc.
    sublinear_tf=True,
    token_pattern=r"(?u)\b\w[\w']+\b",
    min_df=2,                 
    max_df=0.95,              
    max_features=10000        
    )

  
    bi_pipe = Pipeline([
    ("features", FeatureUnion(
        transformer_list=[
            ("uni_vec", uni_vec_for_union),
            ("bi_vec", bi_vec_for_union)
        ],

        transformer_weights={"uni_vec": 1.0, "bi_vec": 0.0}
    )),
    ("classifier", DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ))
    ])

    bi_pipe.memory = mkdtemp()

    bi_grid = {
   
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__max_depth": [5, 6, 8, 12],
    "classifier__min_samples_split": randint(20, 61),
    "classifier__min_samples_leaf": randint(8, 25),
    "classifier__min_impurity_decrease": [0.0, 1e-4, 5e-4, 1e-3],
    "classifier__ccp_alpha": [0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
    "classifier__max_features": [None, "sqrt", "log2"],
    "classifier__max_leaf_nodes": [None, 100, 300, 600],

   
    "features__bi_vec__min_df": [1, 2, 3, 5, 10],
    "features__bi_vec__max_df": [1.0, 0.95, 0.9],
    "features__bi_vec__max_features": [3000, 5000, 10000],
    "features__bi_vec__sublinear_tf": [True], 

    "features__transformer_weights": [
    {"uni_vec":1.0, "bi_vec":0.0},  
    {"uni_vec":1.0, "bi_vec":0.3},
    {"uni_vec":1.0, "bi_vec":0.5},
    {"uni_vec":1.0, "bi_vec":0.7},
    {"uni_vec":1.0, "bi_vec":1.0},
    ]
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