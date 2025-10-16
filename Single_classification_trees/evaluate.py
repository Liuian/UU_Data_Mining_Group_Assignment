import glob as glob
import pandas as pd
import os
import hashlib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def save_unigram_predictions(fnames_test, y_test, y_pred_uni, out_dir="reports"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df = pd.DataFrame({
        "id": fnames_test,
        "true_label": y_test,
        "pred_decision_tree": y_pred_uni
    })
    csv_path = os.path.join(out_dir, "preds.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved unigram predictions to: {csv_path}")

def evaluate_models(uni_model, bi_model, x_test, y_test,fnames_test, labels, out_dir ="reports"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    y_pred_uni = uni_model.best_estimator_.predict(x_test)

    print("Best params:", uni_model.best_params_)
    print("Unigram accuracy:", accuracy_score(y_test, y_pred_uni))
    print(classification_report(y_test, y_pred_uni, target_names=labels))
    save_unigram_predictions(fnames_test,y_test, y_pred_uni, out_dir)
    uni_cm = confusion_matrix(y_test, y_pred_uni)

    ConfusionMatrixDisplay(uni_cm, display_labels=labels).plot(cmap='viridis', colorbar=True)
    plt.title("Unigram Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "unigram_confusion_matrix.png"))
    plt.close()

    y_pred_bi = bi_model.best_estimator_.predict(x_test)

    print("Best params:", bi_model.best_params_)
    print("Bigram accuracy:", accuracy_score(y_test, y_pred_bi))
    print(classification_report(y_test, y_pred_bi, target_names=labels))
    bi_cm = confusion_matrix(y_test, y_pred_bi)

    ConfusionMatrixDisplay(bi_cm, display_labels=labels).plot(cmap='viridis', colorbar=True)
    plt.title("Bigram Confusion Matrix")
    plt.savefig(os.path.join(out_dir, "bigram_confusion_matrix.png"))
    plt.close()

    print(f"\n Confusion matrices saved to '{out_dir}'")

if __name__ == "__main__":
    print("Evaluating models...")