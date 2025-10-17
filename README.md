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

(c) Which external libraries/packages are needed to run your code.  

- Google doc: https://docs.google.com/document/d/1hGF0I7-lEoWSRKTuZkL3C70D-jN3i2rHk9jisJgUOv0/edit?pli=1&tab=t.0#heading=h.140bs3o8so5n

