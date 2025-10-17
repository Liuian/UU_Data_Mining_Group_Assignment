## 1. Multinomial Naive Bayes Classifier
### Requirements

You can install the necessary Python libraries using pip.

```bash
pip install pandas numpy scikit-learn scipy nltk
```

The script will also automatically download the required NLTK data (`stopwords`, `punkt_tab`, `wordnet`) into the script's directory upon first run.

### How to Run

To run the experiment, simply execute the `main_1.py` script from your terminal:

```bash
python ./Logistic_regression_with_Lasso_penalty/main_1.pymain_1.py
```

### Output
- **Logs**: A detailed log of the execution will be saved to `run.log`.
- **Predictions**: The test set predictions will be saved as CSV files inside the `uu_data_mining_group_assignment/reports/` directory (e.g., `uu_data_mining_group_assignment/reports/preds_unigram.csv`).


## 2. Logistic Regression with Lasso (L1) Penalty

### Requirements

You can install the necessary Python libraries using pip.

```bash
pip install pandas numpy scikit-learn scipy nltk
```

The script will also automatically download the required NLTK data (`stopwords`, `punkt_tab`, `wordnet`) into the script's directory upon first run.

### How to Run

To run the experiment, simply execute the `main_1.py` script from your terminal:

```bash
python main_1.py
```

### Output
- **Logs**: A detailed log of the execution will be saved to `run.log`.
- **Predictions**: The test set predictions will be saved as CSV files inside the `uu_data_mining_group_assignment/reports/` directory (e.g., `uu_data_mining_group_assignment/reports/logistic_regression_unigram_count.csv`).


## Single Classification Trees
### How to Run
1. Have Python installed
2. Download the dataset
   Have you data like this:
   
```text
data/
└── negative_polarity/
    ├── truthful_from_Web/
    │   ├── fold1/
    │   ├── fold2/
    │   ├── fold3/
    │   ├── fold4/
    │   └── fold5/
    └── deceptive_from_MTurk/
        ├── fold1/
        ├── fold2/
        ├── fold3/
        ├── fold4/
        └── fold5/


4. Run
   
   python -m pip install -r requirements.txt           #downloads all dependencies
5. Run
   
   python main.py --dataset_root "your_path/negative_polarity"    #"your_path/negative_polarity" replace this with your path to negative_polarity dataset

---

(a) How to run your code (e.g., python main.py).  

(b) Which script(s) produce the key results in your report (e.g., The performance measures reported in Table 2 are produced by train_model.py).

## 4. Gradient Boosting
This project implements a Gradient Boosting to classify text data. It performs hyperparameter tuning using GridSearchCV and evaluates the model on a test set.
Requirements

You can install the necessary Python libraries using pip. It is recommended to use a virtual environment.

This project contains two Jupyter notebooks to train and evaluate **Gradient Boosting** text classifiers on the Deceptive Opinion Spam (Ott et al.) dataset.

- **`GB_UNI.ipynb`** — Gradient Boosting with **unigram** features only.
- **`GB_(UNI_BI).ipynb`** — Gradient Boosting with **unigram + bigram** features.

Both notebooks follow the same experimental protocol: use folds **1–4** for training/tuning and **fold 5** for final evaluation. They report Accuracy, Precision, Recall, F1, and often additional diagnostics.

---

## 1) 

1. **Install dependencies** (ideally in a clean virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

2. **Get the dataset** (Ott et al. “Deceptive Opinion Spam Corpus”, version `op_spam_v1.4` from Kaggle).
   - Place the extracted folder at a path like:
     ```
     /path/to/op_spam_v1.4/
     ```

3. **Launch Jupyter** and run the notebooks in order:
   ```bash
   jupyter lab
   ```
   - Open **`GB_UNI.ipynb`**, run **Kernel → Restart & Run All**.
   - Then open **`GB_(UNI_BI).ipynb`**, run **Kernel → Restart & Run All**.

4. **Find results** in the notebook outputs (metrics printed at the end).

---

## 2) Project Structure

```
.
├── GB_UNI.ipynb
├── GB_(UNI_BI).ipynb
├── requirements.txt
└── README.md   (this file)
```

The notebooks expect the dataset to be available either as an **unzipped folder** or a **.zip** that the code can read/unpack. If a path cell is present (e.g., `data_path`), edit it to point to your actual dataset location.

---


## 3) Environment & Dependencies

- **Python**: 3.9–3.11 recommended
- **Key packages**:
  - `numpy`, `pandas`
  - `scikit-learn`
  - `scipy`
  - `jupyterlab` (or `notebook`)
  - `matplotlib`
  - `ipywidgets` (for optional upload widgets, if used)

Install everything via:
```bash
pip install -r requirements.txt
```

If you prefer conda:
```bash
conda create -n gb_spam python=3.11 -y
conda activate gb_spam
pip install -r requirements.txt
```


## 5.Random Forest
This project implements Random Forest to classify text data. It performs hyperparameter tuning using GridSearchCV and evaluates the model on a test set. Requirements

You can install the necessary Python libraries using pip. It is recommended to use a virtual environment.

This project contains two Jupyter notebooks to train and evaluate **Random Forest** text classifiers on the Deceptive Opinion Spam (Ott et al.) dataset.

- **`Uni_RF.ipynb`** — Random Forest with **unigram** features only.
- **`Uni_bi_RF.ipynb`** — Random Forest with **unigram + bigram** features.

Both notebooks follow the same experimental protocol: use folds **1–4** for training/tuning and **fold 5** for final evaluation. They report Accuracy, Precision, Recall, F1, and may include diagnostics like a confusion matrix or feature stats. Hyperparameters (e.g., number of trees, max features) are selected using **cross‑validation** or **out‑of‑bag (OOB)** performance on the training split.

---

## 1) Quick Start 

1. **Install dependencies** (prefer a clean virtual environment):
   ```bash
   pip install -r requirements_rf.txt
   ```

2. **Get the dataset** (Ott et al. “Deceptive Opinion Spam Corpus”, version `op_spam_v1.4` from Kaggle).
   - Extract and place the folder at a path like:
     ```
     /path/to/op_spam_v1.4/
     ```

3. **Launch Jupyter** and run the notebooks in order:
   ```bash
   jupyter lab
   ```
   - Open **`Uni_RF.ipynb`**, run **Kernel → Restart & Run All**.
   - Open **`Uni_bi_RF.ipynb`**, run **Kernel → Restart & Run All**.

4. **Find results** in the notebook outputs (metrics printed at the end).

---

## 2) Project Structure

```
.
├── Uni_RF.ipynb
├── Uni_bi_RF.ipynb
├── requirements_rf.txt
└── README_RF.md   (this file)
```

> The notebooks expect the dataset to be available either as an **unzipped folder** or a **.zip** that the code can read/unpack. If a path cell exists (e.g., `data_path`), edit it to point to your actual dataset location.

---

## 3) Environment & Dependencies

- **Python**: 3.9–3.11 recommended
- **Key packages**:
  - `numpy`, `pandas`
  - `scikit-learn`
  - `scipy`
  - `jupyterlab` (or `notebook`)
  - `matplotlib`
  - `ipywidgets` (if you use upload widgets)
  - `nbconvert` (for headless execution)

Install everything via:
```bash
pip install -r requirements_rf.txt
```

If you prefer conda:
```bash
conda create -n rf_spam python=3.11 -y
conda activate rf_spam
pip install -r requirements_rf.txt
```

---
