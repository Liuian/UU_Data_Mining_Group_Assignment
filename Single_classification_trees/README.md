A Decision Tree classifier is trained to distinguish between fake and genuine hotel reviews in the dataset. This model utilises unigram and unigram + bigram features for prediction and evaluates performance using accuracy, F1 score, Precision, and Recall scores.

How to Run
1. Have Python installed
2. Download the dataset 
3. Run
   
   python -m pip install -r requirements.txt           #downloads all dependencies
5. Run
   
   python main.py --dataset_root "your_path/negative_polarity"    #"your_path/negative_polarity" replace this with your path to negative_polarity dataset


Notes:
1. Random seeds are fixed, so outputs are reproducible.
2. models/ and reports/ folders are excluded via .gitignore (they can be regenerated)
3. **The code prints and stores all output in reports/output.txt**.






