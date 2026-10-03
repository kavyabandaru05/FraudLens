from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "creditcard_clean.csv"


def clean_data():
    if not RAW_PATH.exists():
        print(f"Dataset not found: {RAW_PATH}")
        return

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_PATH)

    print("Original shape:", df.shape)

    # Remove duplicate rows
    duplicates = df.duplicated().sum()
    df = df.drop_duplicates()

    # Check for missing values
    print("Missing values:", df.isnull().sum().sum())

    # Save cleaned data
    df.to_csv(PROCESSED_PATH, index=False)

    print("Duplicates removed:", duplicates)
    print("Cleaned shape:", df.shape)
    print("Cleaned dataset saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    clean_data()