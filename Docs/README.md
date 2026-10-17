# Breast Cancer Diagnostic Data Mining

## Project summary

This repository contains work on exploratory data analysis and machine-learning classification using the Wisconsin Breast Cancer (Diagnostic) dataset. The project focuses on preprocessing, feature selection, model training (for example Logistic Regression, SVM, Random Forest, and XGBoost), evaluation, and visualization to support diagnostic classification of breast tumors.

## Related article

Breast Cancer Classification using XGBoost
https://www.researchgate.net/publication/378662703_Breast_Cancer_Classification_using_XGBoost#:~:text=employs%20the%20Extreme%20Gradient%20Boosting,diagnostic%29%20dataset

## Dataset

Wisconsin Breast Cancer (Diagnostic) dataset — original source on Kaggle:
https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

## What you'll find in this repository

- Jupyter notebooks and Python scripts (if present) for data preprocessing, feature engineering, model training, evaluation, and visualization.
- Analysis and example model pipelines demonstrating common classification approaches applied to the dataset.

Note: browse the repository root and subfolders to find specific notebooks and scripts used in the analyses.

## Quick usage

1. Clone the repository:
   git clone https://github.com/Teoyed/Breast-Cancer-Diagnostic-Data-Mining.git
2. Create a virtual environment and install dependencies (example):
   python -m venv .venv
   source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
   pip install -r requirements.txt

If there is no requirements.txt, typical packages used are:
- python >=3.8
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- jupyter

3. Start Jupyter and open the notebooks:
   jupyter notebook
