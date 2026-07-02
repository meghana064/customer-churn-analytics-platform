# Customer Churn Prediction

## Project Overview
This project is a professional Machine Learning portfolio template for predicting **customer churn**—whether a customer is likely to stop using a company’s product or service. The repository focuses on a clean, production-minded structure and documentation. (No model is generated in this stage.)

## Problem Statement
Businesses lose revenue when customers leave. Churn prediction uses historical customer attributes and usage/billing patterns to estimate the probability that a customer will churn, enabling proactive retention strategies.

## Objectives
- Build a reproducible ML pipeline for churn prediction (data → features → model → evaluation).
- Provide a lightweight prediction interface for batch or single-customer inference.
- Keep the project organized, auditable, and easy to extend.

## Features
- Clear end-to-end workflow documentation (pipeline, data fields, model options).
- Separated scripts for preprocessing, training, and prediction.
- Dedicated folders for datasets, saved models, notebooks, assets, and docs.
- Streamlit web app with churn prediction, model evaluation, and executive dashboard.

## Machine Learning Workflow
- **Data ingestion**: Collect a churn dataset (e.g., telco churn-style data).
- **Cleaning & validation**: Handle missing values, type conversion, outliers.
- **Feature engineering**: Encode categoricals, scale numeric features, derive helpful ratios/flags.
- **Model training**: Train baseline models and a strong ensemble (Random Forest).
- **Evaluation**: Use metrics such as accuracy, precision/recall, F1, ROC-AUC; inspect confusion matrix.
- **Prediction**: Save a trained model + preprocessing artifacts and run inference on new inputs.
- **Deployment**: Package prediction as a script or a simple Streamlit app.

## Technologies Used
- **Python**
- **pandas**, **numpy** (data handling)
- **scikit-learn** (preprocessing, training, evaluation)
- **joblib** (model serialization)
- **matplotlib** (basic plots)
- **plotly** (interactive executive dashboard charts)
- **streamlit** (web UI for inference, evaluation, and analytics)

## Folder Structure
```text
Customer-Churn-Prediction/
├── app.py
├── train_model.py
├── predict.py
├── preprocess.py
├── model/
│   └── .gitkeep
├── data/
│   └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── assets/
│   └── .gitkeep
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── ML_PIPELINE.md
│   ├── DATA_DESCRIPTION.md
│   └── MODEL_DETAILS.md
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
└── LICENSE
```

## Future Improvements
- Add robust schema validation (e.g., `pandera` or `pydantic`) for input data.
- Add experiment tracking (MLflow) and model registry.
- Hyperparameter tuning and calibration for better probability estimates.
- Add automated tests for preprocessing and prediction (pytest).
- Add CI workflow to lint and run tests.
- Add explainability (permutation importance, SHAP).

## Author
**Meghana Gowda**  
Machine Learning portfolio project: *Customer-Churn-Prediction*

