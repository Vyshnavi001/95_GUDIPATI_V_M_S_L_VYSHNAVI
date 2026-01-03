# Loan Approval Prediction using XGBoost & SHAP

## Project Overview
This project focuses on building an **efficient and explainable loan approval prediction system** using **XGBoost**, a powerful gradient boosting algorithm, combined with **SHAP (SHapley Additive exPlanations)** for model interpretability.

The goal is not only to predict whether a loan should be **approved or rejected**, but also to **explain the reasons behind each decision**, ensuring transparency and fairness in financial decision-making.


## Problem Statement
Banks and financial institutions must evaluate loan applications accurately while maintaining transparency and fairness.  
Traditional models often act as black boxes and fail to explain *why* a loan is approved or rejected.

This project addresses:
- Accurate loan approval prediction
- Feature-level explainability
- Basic fairness analysis across sensitive attributes


## Solution Approach
- **Model**: XGBoost (Gradient Boosting Decision Trees)
- **Explainability**: SHAP (TreeExplainer)
- **Fairness Check**: Group-wise approval comparison
- **Deployment**: FastAPI scoring endpoint


## Dataset
- **Source**: Kaggle – Loan Approval Classification Dataset  
- **Features include**:
  - Applicant Income
  - Coapplicant Income
  - Loan Amount
  - Credit History
  - Education
  - Gender
  - Marital Status
- **Target Variable**:
  - `1` → Loan Approved  
  - `0` → Loan Rejected  


## Tech Stack
- **Python**
- **scikit-learn**
- **XGBoost**
- **SHAP**
- **FastAPI**
- **Pandas, NumPy**
- **Matplotlib / Seaborn**


## Project Workflow
1. Data loading and preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature encoding and scaling  
4. Model training using XGBoost  
5. Model evaluation (Accuracy, Precision, Recall, F1-score)  
6. Explainability using SHAP  
7. Fairness quick-check  
8. API deployment using FastAPI  


## Model Used: XGBoost
XGBoost is an advanced gradient boosting algorithm that:
- Builds trees sequentially
- Focuses on correcting previous errors
- Uses regularization to prevent overfitting
- Performs exceptionally well on structured/tabular data


## Explainability with SHAP
SHAP explains model predictions by assigning **feature-level contribution values**.

### SHAP Outputs:
- **Global Explanation**: Top features influencing loan approval
- **Local Explanation**: Why a specific loan was approved or rejected

### Example Top Features:
1. Credit History  
2. Applicant Income  
3. Loan Amount  
4. Employment Status  
5. Education  


## Fairness Quick-Check
A basic fairness analysis is performed by:
- Comparing loan approval rates across groups
- Attributes checked:
  - Gender
  - Education
  - Marital Status

> Note: This is a preliminary fairness audit and not a full bias mitigation pipeline.


## API Deployment
A **FastAPI** endpoint is provided to score loan applications.

### `/score` Endpoint
**Input**: Applicant details (JSON)  
**Output**:
```json
{
  "prediction": "Approved",
  "probability": 0.83,
  "top_factors": ["Credit_History", "ApplicantIncome", "LoanAmount"]
}
