# Fraud Detection Pipeline (Supervised Learning)

A machine learning project that detects fraudulent transactions in a highly imbalanced dataset (~1% fraud).

## Goal
Build and tune classification models to identify fraud, using SMOTE for class imbalance and evaluating with Precision, Recall and ROC-AUC instead of accuracy.

## Why not accuracy?
A model that predicts "not fraud" for every transaction gets ~99% accuracy but catches zero frauds. So accuracy is misleading here.

## Approach
1. Load data and explore class imbalance
2. Stratified train/test split (SMOTE is never applied to the test set)
3. Scikit-Learn / imblearn pipelines: StandardScaler -> SMOTE -> Model
4. Models: Logistic Regression and Random Forest
5. Hyperparameter tuning with RandomizedSearchCV (scored on PR-AUC)
6. Evaluation: Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC and PR curves
7. Decision threshold tuning, feature importance, model saving with joblib

## Dataset
<Synthetic dataset (fraud_dataset.csv, 50,000 rows, ~1.1% fraud) OR Kaggle Credit Card Fraud Dataset (ULB)>

## Results (test set)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (no SMOTE) | <..> | <..> | <..> | <..> |
| Logistic Regression + SMOTE | <..> | <..> | <..> | <..> |
| Random Forest + SMOTE | <..> | <..> | <..> | <..> |

## Key learnings
- Without SMOTE, recall is low and many frauds are missed.
- SMOTE improves recall, usually at some cost to precision.
- <Which model performed best in your run and why>
- Lowering the decision threshold trades precision for recall.

## How to run
```bash
pip install scikit-learn imbalanced-learn pandas matplotlib seaborn joblib
jupyter notebook fraud_detection_pipeline.ipynb
```
Run all cells from top to bottom.

## Tech stack
Python, Pandas, NumPy, Scikit-Learn, imbalanced-learn, Matplotlib, Seaborn

## Author
Rajan Kumar Tiwari - B.Tech CSE
