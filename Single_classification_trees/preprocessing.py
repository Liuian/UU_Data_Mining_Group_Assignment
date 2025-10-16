from pathlib import Path
import numpy as np
import glob as glob
import pandas as pd


labels = ["Truthful", "Deceptive"]

def load_data(dataset_root, folders):
    review, label, fold, fname = [], [], [], []
    root = Path(dataset_root)
    for label_idx, folder in enumerate(folders):
        for fold_dir in sorted((root/folder).iterdir(), key=lambda p:p.name):
            fold_name = fold_dir.name
            for file in fold_dir.glob("*.txt"):
                review.append(file.read_text())
                label.append(label_idx)
                fold.append(fold_name)
                fname.append(file.name)
    return np.array(review), np.array(label), np.array(fold), np.array(fname) 



if __name__ == "__main__":
    dataset_root = r"C:\Users\ankit\Desktop\Data Mining\Assignment\UU_Data_Mining_Group_Assignment\data\negative_polarity"
    folders = ["truthful_from_Web", "deceptive_from_MTurk"]
    x, y, fold_id, fname = load_data(dataset_root, folders)
    print(f"Loaded {len(x)} reviews from {len(folders)} folders")