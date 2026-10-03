from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "creditcard_clean.csv"
MODEL_PATH = BASE_DIR / "reports" / "fraud_model.pkl"


def train_model():
    """Train, evaluate, and save the fraud detection model."""

    # Check whether the dataset exists
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        print("Run the data loading and cleaning steps first.")
        return

    # Load the processed dataset
    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' target column.")

    if df.empty:
        raise ValueError("The processed dataset is empty.")

    if df["Class"].nunique() != 2:
        raise ValueError("Expected both classes: 0 (legitimate) and 1 (fraud).")

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Handle missing and infinite values
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    X = X.fillna(X.median(numeric_only=True))

    # Ensure all features are numeric
    X = X.select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("No numeric feature columns found.")

    # Split data while preserving fraud proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Build the model pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ])

    # Train
    print("Training fraud detection model...")
    model.fit(X_train, y_train)

    # Evaluate on unseen test data
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred))

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
        )
    )

    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC: {roc_auc:.4f}")

    # Save the trained pipeline
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()