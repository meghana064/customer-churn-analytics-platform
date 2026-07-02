# Data Description (Expected Fields)

This project expects a customer-level dataset where each row represents a customer and the target label indicates churn.

## Core Fields
- **Gender**: Customer gender (e.g., `Male`, `Female`).
- **SeniorCitizen**: Whether the customer is a senior citizen (commonly `0/1`).
- **Partner**: Whether the customer has a partner (`Yes/No`).
- **Dependents**: Whether the customer has dependents (`Yes/No`).
- **Tenure**: Number of months the customer has stayed with the company (integer).
- **PhoneService**: Whether the customer has phone service (`Yes/No`).
- **InternetService**: Type of internet service (e.g., `DSL`, `Fiber optic`, `No`).
- **MonthlyCharges**: Monthly billed amount (numeric).
- **TotalCharges**: Total billed amount to date (numeric; sometimes stored as text in raw exports).
- **Contract**: Contract type (e.g., `Month-to-month`, `One year`, `Two year`).
- **PaymentMethod**: Payment method (e.g., `Electronic check`, `Mailed check`, `Bank transfer`, `Credit card`).
- **Churn**: Target label indicating if the customer churned (`Yes/No`).

## Notes & Expectations
- **Categorical consistency**: Keep category labels consistent (avoid variants like `No internet service` vs `No` unless intentionally modeled).
- **Type handling**: `TotalCharges` may require conversion from string to numeric with missing values handled safely.
- **Optional identifiers**: Some datasets include `customerID`. If present, it should be treated as an identifier and excluded from modeling features.
- **Class imbalance**: Churn datasets are often imbalanced; evaluation should consider precision/recall and ROC-AUC, not accuracy alone.

