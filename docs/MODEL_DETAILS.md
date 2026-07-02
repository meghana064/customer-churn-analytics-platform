# Model Details

This project is designed to support multiple classification models. Below is a high-level overview of common choices for churn prediction.

## Logistic Regression
- **What it is**: A linear model that predicts the probability of churn using a logistic (sigmoid) function.
- **Strengths**:
  - Fast to train and easy to deploy
  - Interpretable coefficients (helpful for explaining drivers of churn)
  - Strong baseline for tabular problems when features are well-prepared
- **Limitations**:
  - Captures only linear relationships unless feature interactions are engineered
  - Sensitive to feature scaling and multicollinearity (depending on setup)

## Decision Tree
- **What it is**: A tree-based model that learns a set of if/then rules to classify churn.
- **Strengths**:
  - Handles nonlinear relationships and feature interactions naturally
  - Works with mixed feature types (after suitable encoding)
  - Interpretable structure (to a point)
- **Limitations**:
  - Prone to overfitting if not carefully regularized (depth, min samples, pruning)
  - Single trees can be unstable to small data changes

## Random Forest
- **What it is**: An ensemble of decision trees trained on bootstrapped samples with randomized feature selection.
- **Strengths**:
  - Strong performance on many tabular datasets
  - More robust than a single decision tree (reduces variance)
  - Captures nonlinear patterns and feature interactions
  - Provides feature importance estimates (useful for analysis)
- **Limitations**:
  - Less interpretable than logistic regression or a single tree
  - Can be heavier for real-time inference (depending on number of trees)

## Why Random Forest is Usually a Strong Choice
Random Forest is often a strong default for churn prediction because churn drivers are frequently **nonlinear** and involve **interactions** (e.g., contract type interacting with monthly charges and tenure). Compared to a single decision tree, Random Forest typically generalizes better and is less sensitive to noise. It also works well without extensive feature engineering beyond clean categorical encoding and basic preprocessing.

