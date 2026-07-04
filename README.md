# Customer Churn Analytics Platform

A Streamlit-based enterprise analytics application for customer churn prediction, retention intelligence, batch scoring, prediction history, executive insights, explainable AI, and model evaluation.

The project is designed for final-year engineering demonstrations, internship portfolios, technical interviews, GitHub hosting, and Streamlit Cloud deployment.

## Features

- Single-customer churn prediction with probability, confidence, risk level, and retention recommendations.
- Batch CSV prediction workflow with validation, summary KPIs, visual analytics, and downloadable results.
- Executive dashboard for churn distribution, revenue exposure, segment analysis, and business KPIs.
- Customer search and prediction history with filters, badges, progress bars, and clear-session controls.
- Executive business insights with retention strategy, revenue risk, and management takeaways.
- Explainable AI page for model-level and customer-level interpretation.
- Model evaluation page with performance metrics, confusion matrix, ROC curve, and model comparison.
- Professional responsive UI with custom theme, cards, charts, navigation, and deployment-ready Streamlit config.

## Architecture

```text
Streamlit UI (app.py)
|-- UI theme and reusable components (ui_theme.py)
|-- Single-customer prediction (predict.py)
|-- Batch scoring and validation (batch_predict.py)
|-- Session prediction history (prediction_history.py)
|-- Executive analytics (executive_insights.py)
|-- PDF reporting (pdf_report.py)
|-- Data preprocessing and model training utilities (preprocess.py, train_model.py)
|-- Local artifacts (data/, model/)
```

The deployed app uses local, repository-tracked model artifacts and dataset files. No external database, API key, or secret is required.

## Technologies Used

- Python
- Streamlit
- pandas
- numpy
- scikit-learn
- joblib
- Plotly
- matplotlib
- ReportLab

## Installation

1. Clone the repository:

```bash
git clone https://github.com/meghana064/customer-churn-analytics-platform.git
cd customer-churn-analytics-platform
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Locally

Run the Streamlit app from the project root:

```bash
streamlit run app.py
```

The app entry point is `app.py`, which is also the file Streamlit Cloud should use.

## Required Deployment Files

The following files are required for the deployed app to work:

- `app.py`
- `requirements.txt`
- `data/Telco-Customer-Churn.csv`
- `model/churn_model.pkl`
- `model/label_encoders.pkl`
- `preprocess.py`
- `train_model.py`
- `predict.py`
- `batch_predict.py`
- `executive_insights.py`
- `prediction_history.py`
- `pdf_report.py`
- `ui_theme.py`
- `.streamlit/config.toml`

## Model Information

- Trained model artifact: `model/churn_model.pkl`
- Label encoder artifact: `model/label_encoders.pkl`
- Training utility: `train_model.py`
- Inference utility: `predict.py`
- The app loads the saved model artifacts at runtime and does not retrain the model during deployment.
- The current pipeline supports Logistic Regression, Decision Tree, and Random Forest evaluation, with the selected trained artifact persisted in `model/`.

## Dataset

- Dataset file: `data/Telco-Customer-Churn.csv`
- Dataset type: IBM Telco Customer Churn-style customer subscription data
- Target column: `Churn`
- Identifier column: `customerID`
- The dataset is required for executive dashboards, model evaluation, insights, and explainability pages.

## Streamlit Cloud Deployment

1. Push this project to GitHub.
2. Open Streamlit Cloud.
3. Select the GitHub repository.
4. Set the main file path to:

```text
app.py
```

5. Deploy the app.

No secrets are required for the current local model and dataset workflow. If secrets are added later, store them in `.streamlit/secrets.toml` locally and configure them through Streamlit Cloud secrets.

## Screenshots

Add screenshots after deployment:

- Executive Dashboard
- Customer Prediction
- Batch Prediction
- Customer Search and History
- Executive Business Insights
- Explainable AI
- Model Evaluation

## Future Improvements

- Add automated UI regression tests for Streamlit pages.
- Add CI checks for linting, import validation, and smoke tests.
- Add model monitoring once live user data is connected.
- Add optional authentication for production business use.
- Add explainability methods such as SHAP if dependency size is acceptable for deployment.

## Folder Structure

```text
Customer-Churn-Prediction/
|-- app.py
|-- batch_predict.py
|-- executive_insights.py
|-- pdf_report.py
|-- prediction_history.py
|-- ui_theme.py
|-- preprocess.py
|-- train_model.py
|-- predict.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|-- .streamlit/
|   |-- config.toml
|-- data/
|   |-- Telco-Customer-Churn.csv
|-- model/
|   |-- churn_model.pkl
|   |-- label_encoders.pkl
|-- docs/
|   |-- DATA_DESCRIPTION.md
|   |-- DATA_SOURCE.md
|   |-- ML_PIPELINE.md
|   |-- MODEL_DETAILS.md
|   |-- PROJECT_OVERVIEW.md
|-- assets/
|-- notebooks/
|-- LICENSE
```

## Author

**Meghana Gowda M**

Customer Churn Analytics Platform - machine learning, analytics, and Streamlit deployment project.

## License

This project is licensed under the terms included in `LICENSE`.

