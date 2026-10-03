FraudLens — Credit Card Fraud Detection

Overview
FraudLens is a data analytics and machine learning project designed to detect potentially fraudulent credit card transactions and analyze model performance using Power BI.

Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- PostgreSQL (planned)
- Power BI
- Git and GitHub

Key Features

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Fraud detection using Random Forest
- Model evaluation using ROC-AUC, precision, and recall
- Threshold comparison to understand fraud detection trade-offs
- Interactive Power BI dashboard for visualizing results

Project Structure

- "data/" — Raw and processed datasets
- "src/" — Python source code
- "reports/" — Model, evaluation results, and charts
- "main.py" — Main project execution file

Model Results

- ROC-AUC: 0.9447
- Test transactions: 56,746
- Actual fraud cases in the test set: 95

Dashboard
The Power BI dashboard compares fraud recall, precision, and fraud detection outcomes at different classification thresholds.

Disclaimer
This project is for educational and portfolio purposes. Model predictions should not be treated as definitive proof of fraud.

Author
Kavya chowdhary

FraudLens is a data analytics and machine learning project designed to detect potentially fraudulent credit card transactions and analyze model performance using Power BI.

Features

- Data cleaning and preprocessing using Python and Pandas
- Fraud detection using Random Forest
- Model evaluation using ROC-AUC, precision, and recall
- Threshold comparison to analyze fraud detection performance
- Interactive Power BI dashboard for visualization

Tech Stack

- Python, Pandas, Scikit-learn
- Matplotlib
- Power BI
- Git and GitHub

Model Results

- Test transactions: 56,746
- ROC-AUC: 0.9447
- Fraud recall at threshold 0.30: 73.68%
- Fraud precision at threshold 0.30: 95.89%

Project Structure

- "src/" — Python scripts for cleaning, analysis, and modeling
- "reports/" — Evaluation results and visualizations
- "requirements.txt" — Python dependencies

Dataset

The credit card transaction dataset is not included because of its large size. Obtain the dataset separately and place it in the required data directory before running the project.

How to Run

1. Install the dependencies:
   "pip install -r requirements.txt"
2. Place the dataset in "data/raw/creditcard.csv".
3. Run the main script:
   "python main.py"

Dashboard

The Power BI dashboard compares fraud recall, precision, and fraud outcomes across different classification thresholds.
