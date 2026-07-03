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
git clone <your-repository-url>
cd Customer-Churn-Prediction
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

