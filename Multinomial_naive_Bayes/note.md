1. 資料準備
    - 使用提供的飯店評論資料（800 篇，negative truthful vs negative deceptive）。
    - folds 1–4 (640 reviews) → training + hyperparameter tuning
    - fold 5 (160 reviews) → final test set（只能用來報最終模型效能）。
    - 做文字前處理（tokenization, lowercase, stopwords, stemming/lemmatization 可自行決定），並轉換成特徵（Count 或 TF-IDF）。
2. Multinomial Naive Bayes
    - 模型：sklearn.naive_bayes.MultinomialNB
    - 要做的事：
        - 設定/調整超參數 alpha（平滑參數），測試不同數值（例如 0.1, 0.5, 1.0）。
        - 做 特徵選擇（例如挑 top-k 最有用的詞；k = 500, 1000, 2000…）。
        - 在 folds 1–4 上做 cross-validation 找出最佳超參數組合。
        - 用最佳設定在 folds 1–4 retrain 模型，再到 fold 5 測試效能。
        - 報告 accuracy, precision, recall, F1。
    - 做兩組實驗：
        1. 只用 unigram 特徵
        2. 用 unigram + bigram 特徵
3. 找出「最重要的 5 個詞」
    - Naive Bayes：計算 log-odds 或直接比較 feature_log_prob_，找出最能指向 fake 的前 5 個詞，和最能指向 genuine 的前 5 個詞。
    - Logistic Regression：用 coef_，係數大的詞 → 指向 fake，係數小的詞 → 指向 genuine。
4. 最後產出
    - 這兩個模型的最終測試效能（fold 5 的 accuracy, precision, recall, F1）。
    - 與其他組員模型的結果一起比較（誰效能最好，差異是否顯著）。
    - 解釋最重要的 5 個詞（fake / genuine）。
    - 把你的程式碼清楚分開存放，例如 naive_bayes.py, logistic_regression.py。


- For main_1.py
```python
"""
This script implements and evaluates a Multinomial Naive Bayes classifier for
deceptive opinion spam detection. It follows a two-stage experimental design
to compare the performance of a model using only unigram features against a
model using a combination of unigram and bigram features.

Experimental Design:

The `main` function orchestrates two independent experiments by calling the
`run_experiment` function twice:

1.  Unigram Only (`run_experiment((1,1), ...)`):
    - A `CountVectorizer` is configured with `ngram_range=(1,1)`.
    - It builds a candidate vocabulary consisting solely of unigrams (single words)
      from the training data.
    - `GridSearchCV` then searches for the best `max_features` by selecting the
      top N most frequent unigrams from this candidate pool.

2.  Unigram + Bigram (`run_experiment((1,2), ...)`):
    - A new `CountVectorizer` is configured with `ngram_range=(1,2)`.
    - It builds a new, mixed candidate vocabulary containing *both* unigrams
      and bigrams from the training data.
    - `GridSearchCV` again searches for the best `max_features`. However, in this
      run, it selects the top N features from the *mixed* pool, where unigrams
      and bigrams compete for inclusion based on their overall frequency.

Key Distinction:
The core difference lies in how the final features are selected:

-   **Identical Candidate Pool for Unigrams:** Before hyperparameter tuning,
    the pool of *all possible* unigram features is identical for both
    experiments because they both process the same training data.

-   **Different Competitive Environments:**
    - In the first experiment, unigrams only compete against other unigrams
      for a spot in the final `max_features` set.
    - In the second experiment, unigrams must compete against bigrams for
      those same spots.

As a result, the final set of unigram features chosen for the second model is
not guaranteed to be the same as the set chosen for the first model. This
design differs from a `FeatureUnion` approach, where a pre-selected unigram
vocabulary is explicitly combined with a separately generated bigram vocabulary.
"""
```