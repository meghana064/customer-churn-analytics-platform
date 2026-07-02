"""
Machine learning training pipeline for Customer Churn Prediction.

This module loads and preprocesses data using ``preprocess.py``, trains multiple
classification models, evaluates them, selects the best performer, and persists
trained artifacts to the ``model/`` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

from preprocess import (
    clean_dataset,
    encode_features,
    inspect_dataset,
    load_dataset,
    split_features_target,
    train_test_split_data,
)

DEFAULT_DATA_PATH = Path("data") / "Telco-Customer-Churn.csv"
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"

IDENTIFIER_COLUMNS: Tuple[str, ...] = ("customerID",)


def _drop_identifier_columns(X: pd.DataFrame) -> pd.DataFrame:
    """
    Remove identifier columns that should not be used as model features.

    Args:
        X: Feature matrix.

    Returns:
        Feature matrix without identifier columns.
    """
    columns_to_drop = [col for col in IDENTIFIER_COLUMNS if col in X.columns]
    if not columns_to_drop:
        return X
    return X.drop(columns=columns_to_drop)


def _build_models() -> Dict[str, Any]:
    """
    Create the candidate classifiers for churn prediction.

    Returns:
        Mapping of model names to untrained estimator instances.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a trained classifier on the test set.

    Args:
        model: Trained scikit-learn estimator with ``predict`` support.
        X_test: Test feature matrix.
        y_test: Test target labels.

    Returns:
        Dictionary containing accuracy, precision, recall, f1_score, and roc_auc.

    Raises:
        ValueError: If evaluation fails due to invalid predictions or probabilities.
    """
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception as exc:
        raise ValueError(f"Model evaluation failed: {exc}") from exc

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Train all candidate models and evaluate them on the test set.

    Args:
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target labels.
        y_test: Test target labels.

    Returns:
        A tuple of:
          - metrics dictionary keyed by model name
          - fitted models dictionary keyed by model name

    Raises:
        ValueError: If training or evaluation fails for any model.
    """
    metrics: Dict[str, Dict[str, float]] = {}
    fitted_models: Dict[str, Any] = {}

    for model_name, model in _build_models().items():
        try:
            model.fit(X_train, y_train)
            fitted_models[model_name] = model
            metrics[model_name] = evaluate_model(model, X_test, y_test)
        except Exception as exc:
            raise ValueError(
                f"Failed to train or evaluate model '{model_name}': {exc}"
            ) from exc

    return metrics, fitted_models


def select_best_model(metrics: Dict[str, Dict[str, float]]) -> str:
    """
    Select the best model using ROC-AUC, with F1 Score as the tiebreaker.

    Args:
        metrics: Evaluation metrics keyed by model name.

    Returns:
        Name of the best-performing model.

    Raises:
        ValueError: If no metrics are provided.
    """
    if not metrics:
        raise ValueError("No model metrics available for selection.")

    return max(
        metrics,
        key=lambda name: (metrics[name]["roc_auc"], metrics[name]["f1_score"]),
    )


def display_comparison_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Build a comparison table for all trained models.

    Args:
        metrics: Evaluation metrics keyed by model name.

    Returns:
        DataFrame with columns:
        Model, Accuracy, Precision, Recall, F1 Score, ROC-AUC
    """
    rows: List[Dict[str, Any]] = []
    for model_name, model_metrics in metrics.items():
        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(model_metrics["accuracy"], 4),
                "Precision": round(model_metrics["precision"], 4),
                "Recall": round(model_metrics["recall"], 4),
                "F1 Score": round(model_metrics["f1_score"], 4),
                "ROC-AUC": round(model_metrics["roc_auc"], 4),
            }
        )

    return pd.DataFrame(rows)


def save_artifacts(model: Any, encoders: Dict[str, Any]) -> None:
    """
    Persist the best model and label encoders to disk.

    Args:
        model: Trained best-performing model.
        encoders: Dictionary of fitted label encoders.

    Raises:
        OSError: If model artifacts cannot be written to disk.
    """
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
    except Exception as exc:
        raise OSError(f"Failed to save model artifacts: {exc}") from exc


def run_training_pipeline(
    csv_path: str | Path = DEFAULT_DATA_PATH,
) -> Dict[str, Any]:
    """
    Execute the full churn model training workflow.

    Steps:
      1. Load dataset
      2. Inspect dataset
      3. Clean dataset
      4. Encode categorical features
      5. Split features and target
      6. Perform train/test split
      7. Train and evaluate models
      8. Select best model
      9. Save artifacts

    Args:
        csv_path: Path to the churn dataset CSV.

    Returns:
        Structured dictionary containing:
          - best_model
          - metrics
          - training_samples
          - testing_samples
          - comparison_table

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If preprocessing or training fails.
        OSError: If model artifacts cannot be saved.
    """
    try:
        raw_df = load_dataset(csv_path)
        inspection = inspect_dataset(raw_df)
        cleaned_df = clean_dataset(raw_df)
        encoded_df, encoders = encode_features(cleaned_df)
        X, y = split_features_target(encoded_df)
        X = _drop_identifier_columns(X)

        X_train, X_test, y_train, y_test = train_test_split_data(X, y)
        metrics, fitted_models = train_and_evaluate_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )

        best_model_name = select_best_model(metrics)
        best_model = fitted_models[best_model_name]
        comparison_table = display_comparison_table(metrics)

        save_artifacts(best_model, encoders)

        return {
            "best_model": best_model_name,
            "metrics": metrics,
            "training_samples": int(len(X_train)),
            "testing_samples": int(len(X_test)),
            "comparison_table": comparison_table,
            "inspection": inspection,
        }
    except FileNotFoundError:
        raise
    except (TypeError, ValueError, OSError):
        raise
    except Exception as exc:
        raise ValueError(f"Training pipeline failed: {exc}") from exc


def main() -> Dict[str, Any]:
    """
    Run the training pipeline and print a summary to stdout.

    Returns:
        Structured training summary dictionary.
    """
    summary = run_training_pipeline()

    print("Dataset inspection complete.")
    print("\nModel Comparison:")
    print(summary["comparison_table"].to_string(index=False))
    print(f"\nBest Model: {summary['best_model']}")
    print(f"Training Samples: {summary['training_samples']}")
    print(f"Testing Samples: {summary['testing_samples']}")
    print(f"\nSaved model to: {MODEL_PATH}")
    print(f"Saved encoders to: {ENCODERS_PATH}")

    return summary


if __name__ == "__main__":
    main()
