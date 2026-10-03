from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "creditcard_clean.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"


def analyze_data():
    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    print("DATASET SUMMARY")
    print("-" * 40)
    print("Total transactions:", len(df))
    print("Total columns:", len(df.columns))

    print("\nTRANSACTION CLASS COUNTS")
    class_counts = df["Class"].value_counts().reindex([0, 1], fill_value=0)
    class_counts.index = ["Legitimate", "Fraudulent"]
    print(class_counts)

    fraud_count = int((df["Class"] == 1).sum())
    fraud_percentage = fraud_count / len(df) * 100 if len(df) else 0

    print(f"\nFraud percentage: {fraud_percentage:.4f}%")

    print("\nTRANSACTION AMOUNT SUMMARY")
    print(df["Amount"].describe())

    print("\nAVERAGE AMOUNT BY CLASS")
    print(df.groupby("Class")["Amount"].mean())

    # Chart 1: Transaction class distribution
    plt.figure(figsize=(7, 5))
    class_counts.plot(kind="bar")
    plt.title("Legitimate vs Fraudulent Transactions")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of Transactions")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "transaction_distribution.png", dpi=150)
    plt.close()

    # Chart 2: Transaction amount distribution
    plt.figure(figsize=(7, 5))
    plt.hist(df["Amount"], bins=50)
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "amount_distribution.png", dpi=150)
    plt.close()

    # Chart 3: Average amount by transaction class
    average_amount = df.groupby("Class")["Amount"].mean()
    average_amount.index = ["Legitimate", "Fraudulent"]

    plt.figure(figsize=(7, 5))
    average_amount.plot(kind="bar")
    plt.title("Average Amount by Transaction Type")
    plt.xlabel("Transaction Type")
    plt.ylabel("Average Amount")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "average_amount_by_class.png", dpi=150)
    plt.close()

    print("\nCharts saved successfully in the reports folder:")
    print("1. transaction_distribution.png")
    print("2. amount_distribution.png")
    print("3. average_amount_by_class.png")


if __name__ == "__main__":
    analyze_data()