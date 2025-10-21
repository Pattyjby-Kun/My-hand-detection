# train_gesture.py
import argparse, joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def parse_args():
    ap = argparse.ArgumentParser(description="Train gesture classifier from CSV features")
    ap.add_argument("--csv", type=str, default="my_dataset.csv", help="Path to dataset CSV")
    ap.add_argument("--out", type=str, default="gesture_model.pkl", help="Output model path")
    return ap.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    # columns: timestamp, label, finger_count, tip1_x_norm ... tip5_theta
    y = df["label"].values
    # ใช้ทุกฟีเจอร์ยกเว้น timestamp/label
    X = df.drop(columns=["timestamp", "label"]).values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # pipeline: Standardize + SVM (RBF)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=4.0, gamma="scale", probability=True))
    ])

    # quick holdout + cv
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    pipe.fit(Xtr, ytr)

    print("\n=== Validation on holdout ===")
    ypred = pipe.predict(Xte)
    print(classification_report(yte, ypred, target_names=le.classes_))
    print("\nConfusion matrix:\n", confusion_matrix(yte, ypred))

    # 5-fold CV score
    scores = cross_val_score(pipe, X, y_enc, cv=5)
    print(f"\n5-fold CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    # save model + label encoder
    joblib.dump({"pipeline": pipe, "label_encoder": le}, args.out)
    print(f"\n✅ Saved model to: {args.out}")

if __name__ == "__main__":
    main()
