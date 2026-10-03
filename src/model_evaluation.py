from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "creditcard_clean.csv"
MODEL_PATH = BASE_DIR / "reports" / "fraud_model.pkl"
REPORT_PATH = BASE_DIR / "reports" / "threshold_comparison.csv"

THRESHOLDS = [0.3, 0.4, 0.5]


def evaluate_model():
    # Check whether the dataset exists
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    # Check whether the trained model exists
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Please train the model first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        print("Error: 'Class' column not found in dataset.")
        return

    if df.empty:
        print("Error: Dataset is empty.")
        return

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Handle infinite and missing values consistently
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(X.median(numeric_only=True))
    X = X.select_dtypes(include=["number"])

    # Use the same train-test split as model.py
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Loading trained model...")
    model = joblib.load(MODEL_PATH)

    # IMPORTANT: Calculate y_prob BEFORE the threshold loop
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("FRAUD DETECTION MODEL EVALUATION")
    print("=" * 50)

    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Test transactions: {len(y_test)}")
    print(f"Actual fraud cases: {int(y_test.sum())}")

    # Store threshold comparison results
    results = []

    # Evaluate different decision thresholds
    for threshold in THRESHOLDS:
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()

        fraud_recall = tp / (tp + fn) if (tp + fn) else 0
        fraud_precision = tp / (tp + fp) if (tp + fp) else 0

        print("\n" + "=" * 50)
        print(f"DECISION THRESHOLD: {threshold}")
        print("=" * 50)

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                labels=[0, 1],
                target_names=["Legitimate", "Fraud"],
                zero_division=0,
            )
        )

        print("Fraud Detection Summary:")
        print(f"True Positives  (fraud detected): {tp}")
        print(f"False Negatives (fraud missed):   {fn}")
        print(f"False Positives (false alarms):   {fp}")
        print(f"True Negatives  (legitimate):     {tn}")
        print(f"Fraud Recall:    {fraud_recall:.2%}")
        print(f"Fraud Precision: {fraud_precision:.2%}")

        # Collect results for the CSV report
        results.append({
            "threshold": threshold,
            "true_positives": int(tp),
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "fraud_recall": round(fraud_recall, 4),
            "fraud_precision": round(fraud_precision, 4),
        })

    # Step 3: Save the CSV AFTER the loop
    results_df = pd.DataFrame(results)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORT_PATH, index=False)

    print("\n" + "=" * 50)
    print("THRESHOLD COMPARISON")
    print("=" * 50)
    print(results_df.to_string(index=False))

    print(f"\nThreshold comparison saved to: {REPORT_PATH}")


if __name__ == "__main__":
    evaluate_model()