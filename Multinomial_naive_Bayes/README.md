# Multinomial Naive Bayes Classifier

This project implements a Multinomial Naive Bayes classifier to classify text data. It performs hyperparameter tuning using GridSearchCV and evaluates the model on a test set.

## Requirements

You can install the necessary Python libraries using pip. It is recommended to use a virtual environment.

```bash
pip install pandas numpy scikit-learn scipy nltk
```

The script will also automatically download the required NLTK data (`stopwords`, `punkt_tab`, `wordnet`) into the script's directory upon first run.

## How to Run

To run the experiment, simply execute the `main_1.py` script from your terminal:

```bash
python main_1.py
```

### Output
- **Logs**: A detailed log of the execution will be saved to `run.log`.
- **Predictions**: The test set predictions will be saved as CSV files inside the `reports/` directory (e.g., `reports/preds_unigram.csv`).
