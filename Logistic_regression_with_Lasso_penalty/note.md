1. 資料準備
    - 使用提供的飯店評論資料（800 篇，negative truthful vs negative deceptive）。
    - folds 1–4 (640 reviews) → training + hyperparameter tuning
    - fold 5 (160 reviews) → final test set（只能用來報最終模型效能）。
    - 做文字前處理（tokenization, lowercase, stopwords, stemming/lemmatization 可自行決定），並轉換成特徵（Count 或 TF-IDF）。
3. Logistic Regression with Lasso Penalty
    - 模型：sklearn.linear_model.LogisticRegression(penalty="l1")
    - 要做的事：
        - 調整超參數 C = 1/λ（正則化強度），測試不同數值（例如 0.001, 0.01, 0.1, 1, 10）。
        - 在 folds 1–4 上用 cross-validation 找出最佳 C。
        - Retrain 最佳模型，再到 fold 5 測試。
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





