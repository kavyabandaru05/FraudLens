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