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
3. The code prints and stores all output in reports/output.txt.

reports/output.txt results:

Train set size: 640 | Test set size: 160
Training unigram model....
Fitting 4 folds for each of 30 candidates, totalling 120 fits
Training bigram model....
Fitting 4 folds for each of 30 candidates, totalling 120 fits
Saved models to models

Done. Training complete. Models to models
Best params: {'classifier__ccp_alpha': 0.0001, 'classifier__criterion': 'entropy', 'classifier__max_depth': 5, 'classifier__min_samples_leaf': 3, 'classifier__min_samples_split': 30, 'vectorizer__max_df': 1.0, 'vectorizer__max_features': 10000, 'vectorizer__min_df': 2}
Unigram accuracy: 0.68125
              precision    recall  f1-score   support

    Truthful       0.67      0.70      0.69        80
   Deceptive       0.69      0.66      0.68        80

    accuracy                           0.68       160
   macro avg       0.68      0.68      0.68       160
weighted avg       0.68      0.68      0.68       160

Best params: {'classifier__ccp_alpha': 0.0, 'classifier__criterion': 'log_loss', 'classifier__max_depth': 5, 'classifier__min_samples_leaf': 10, 'classifier__min_samples_split': 22, 'features__bi_vec__max_df': 1.0, 'features__bi_vec__max_features': 20000, 'features__bi_vec__min_df': 2}
Bigram accuracy: 0.66875
              precision    recall  f1-score   support

    Truthful       0.66      0.70      0.68        80
   Deceptive       0.68      0.64      0.66        80

    accuracy                           0.67       160
   macro avg       0.67      0.67      0.67       160
weighted avg       0.67      0.67      0.67       160


 Confusion matrices saved to 'reports'

Done. Evaluation complete. Confusion matrices saved to reports
Top 5 fake terms pointing towards fake reviews: 

feature  importance  correlation
chicago    0.470411     0.418747
decided    0.085466     0.219459
 turned    0.081336     0.159495
   east    0.067148     0.047488
finally    0.065181     0.205610
=====================================================================

Top 5 genuine terms pointing towards genuine reviews: 

feature  importance  correlation
 called    0.027689    -0.050063
Complete. End of script





