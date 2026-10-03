FraudLens — Credit Card Fraud Detection

Overview

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
