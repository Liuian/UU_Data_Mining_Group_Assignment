if __name__ == "__main__":
    import numpy as np
    import sys
    from preprocessing import load_data, labels
    from train_models import train_models
    from evaluate import evaluate_models
    from features import get_features

    dataset_root = r"C:\Users\ankit\Desktop\Data Mining\Assignment\UU_Data_Mining_Group_Assignment\data\negative_polarity"
    folders = ["truthful_from_Web", "deceptive_from_MTurk"]
    x, y, fold_id = load_data(dataset_root, folders)
    test_folder = "fold5"
    train_mask = fold_id != test_folder

    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[~train_mask], y[~train_mask]
    
    sys.stdout = open("reports/output.txt", "w", encoding="utf-8")
    print(f"Train set size: {len(x_train)} | Test set size: {len(x_test)}")

    fold_train = np.array([int(f.replace("fold", "")) for f in fold_id[train_mask]])
    uni_model, bi_model = train_models(x_train, y_train, fold_train, save_dir="models")

    print("\nDone. Training complete. Models to models")

    evaluate_models(uni_model, bi_model, x_test, y_test, labels, out_dir="reports")
    print("\nDone. Evaluation complete. Confusion matrices saved to reports")

    get_features(uni_model, x_train, y_train, k=5)

    print("Complete. End of script")