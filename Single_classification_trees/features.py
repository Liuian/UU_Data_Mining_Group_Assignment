
import numpy as np
import glob as glob
import pandas as pd

from scipy.stats import pointbiserialr


def get_features(uni_model, x_train, y_train, k = 5):
    #extraction of trained components
    vec = uni_model.best_estimator_.named_steps["vectorizer"]
    tree = uni_model.best_estimator_.named_steps["classifier"]

    #Feature names and importance scores
    features = np.array(vec.get_feature_names_out())
    importance = tree.feature_importances_

    #Getting document-term matrix for correlation analysis
    x_vec =vec.transform(x_train)
    X_bin = (x_vec > 0).astype(int)
    y_arr = np.asarray(y_train).ravel()

    #computing correlation of each feature with the label (fake=1, genuine=0)
    correlations = []
    for i in range(X_bin.shape[1]):
        corr, _ = pointbiserialr(X_bin[:, i].toarray().ravel(), y_arr)
        correlations.append(corr)
    correlations = np.nan_to_num(correlations)

    #Combine feature info into a dataframe
    df = pd.DataFrame({
        "feature": features,
        "importance": importance,
        "correlation": correlations
    })

    #filter only features that actually appearin the tree (nonzero importance)
    df = df[df["importance"] > 0]

    #sort by importance
    df_sorted = df.sort_values(by="importance", ascending=False)

    #Separate those correlated with fake and genuine
    fake_terms = df_sorted[df_sorted["correlation"] > 0].head(k)
    genuine_terms = df_sorted[df_sorted["correlation"] < 0].head(k)

    cols = ["feature", "importance", "correlation"]

    print("Top 5 fake terms pointing towards fake reviews: \n")
    print(fake_terms[cols].reset_index(drop=True).to_string(index=False))
    print("=====================================================================")
    print("\nTop 5 genuine terms pointing towards genuine reviews: \n")
    print(genuine_terms[cols].reset_index(drop=True).to_string(index=False))

    if __name__ == "__main__":
        print("Feature evaluuation...")