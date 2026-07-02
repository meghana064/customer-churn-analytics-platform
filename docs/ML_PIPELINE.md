# ML Pipeline

This document describes a typical end-to-end Machine Learning workflow for a Customer Churn Prediction system.

## Data Collection
- Gather customer demographics, account info, service subscriptions, billing details, and churn labels.
- Sources can include CRM systems, billing databases, support ticket systems, and product usage logs.
- Ensure data access is compliant with privacy and security requirements.

## Data Cleaning
- Handle missing values (impute or remove, depending on impact and context).
- Convert data types (e.g., `TotalCharges` sometimes loads as text).
- Remove duplicates and fix inconsistent categorical values.
- Validate ranges (e.g., tenure cannot be negative).

## Feature Engineering
- Encode categorical variables (one-hot encoding for services, contract types, payment methods).
- Scale/transform numeric features if needed.
- Create useful derived signals, such as:
  - Long-tenure vs short-tenure flags
  - Auto-pay indicator
  - Monthly-to-total charge relationships
- Keep feature steps reproducible and applied identically in training and inference.

## Model Training
- Split data into training/validation/test sets.
- Train baseline models first (e.g., Logistic Regression, Decision Tree).
- Train stronger ensemble models (e.g., Random Forest).
- Use cross-validation and hyperparameter tuning as a later improvement.

## Evaluation
- Evaluate using classification metrics:
  - Accuracy
  - Precision / Recall
  - F1-score
  - ROC-AUC (recommended for churn probability ranking)
- Review confusion matrix to understand false positives vs false negatives.
- Consider business costs:
  - False negative: missing a customer who will churn
  - False positive: spending retention budget unnecessarily

## Prediction
- Persist preprocessing artifacts and the trained model (e.g., via `joblib`).
- For new customers:
  - Apply the exact same preprocessing pipeline
  - Output churn probability and predicted class
- Provide batch prediction for CSVs and single-customer prediction via structured input.

## Deployment
- Options include:
  - A small Streamlit app for demos and stakeholder review
  - A REST API (FastAPI/Flask) for production integration
  - Scheduled batch scoring pipelines (e.g., daily churn risk scoring)
- Monitor performance drift and retrain periodically as data changes.

