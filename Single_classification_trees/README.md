A Decision Tree classifier is trained to distinguish between fake and genuine hotel reviews in the dataset. This model utilises unigram and unigram + bigram features for prediction and evaluates performance using accuracy, F1 score, Precision, and Recall scores.

How to Run
1. Have Python installed
2. Download the dataset 
3. Run
   
   python -m pip install -r requirements.txt           #downloads all dependencies
5. Run
   
   python main.py --dataset_root "your_path/negative_polarity"    #"your_path/negative_polarity" replace this with your path to negative_polarity dataset


Files:
| File                   | Description                                                                                                                                  |
| ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **`main.py`**          | Runs the full pipeline: loads data, trains models, evaluates results, and prints top terms.                                                  |
| **`preprocessing.py`** | Loads and prepares the dataset, splits folds for training and testing.                                          |
| **`train_models.py`**  | Defines and trains the **Decision Tree classifier** (unigram and bigram versions) using RandomizedSearchCV with GroupKFold cross-validation. |
| **`evaluate.py`**      | Evaluates trained models on the test set, computes accuracy, precision, recall, F1-score, and saves confusion matrix plots and reports.      |
| **`features.py`**      | Extracts and displays the **five most important terms** indicating fake and genuine reviews based on feature importance and correlation.     |
| **`requirements.txt`** | Lists all Python dependencies required to run the project.                                                                                   |
| **`README.md`**        | Provides setup instructions, project overview, and usage details.                                                                            |
| **`.gitignore`**       | Ensures large or unnecessary files (like `data/`, `models/`, `reports/`) are not uploaded to GitHub.                                         |



Notes:
1. Random seeds are fixed, so outputs are reproducible.
2. models/ and reports/ folders are excluded via .gitignore (they can be regenerated)
3. **The code prints and stores all output in reports/output.txt**.






