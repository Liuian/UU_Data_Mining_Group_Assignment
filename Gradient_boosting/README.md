Gradient Boosting
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

