Random Forest
This project implements Random Forest to classify text data. It performs hyperparameter tuning using GridSearchCV and evaluates the model on a test set. Requirements

You can install the necessary Python libraries using pip. It is recommended to use a virtual environment.

This project contains two Jupyter notebooks to train and evaluate **Random Forest** text classifiers on the Deceptive Opinion Spam (Ott et al.) dataset.

- **`Uni_RF.ipynb`** — Random Forest with **unigram** features only.
- **`Uni_bi_RF.ipynb`** — Random Forest with **unigram + bigram** features.

Both notebooks follow the same experimental protocol: use folds **1–4** for training/tuning and **fold 5** for final evaluation. They report Accuracy, Precision, Recall, F1, and may include diagnostics like a confusion matrix or feature stats. Hyperparameters (e.g., number of trees, max features) are selected using **cross‑validation** or **out‑of‑bag (OOB)** performance on the training split.

---

## 1) Quick Start (TL;DR)

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
