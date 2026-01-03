# Loan Approval Prediction using XGBoost & SHAP

## Project Overview
This project focuses on building an efficient and explainable loan approval prediction system using XGBoost, a powerful gradient boosting algorithm, combined with SHAP (SHapley Additive exPlanations) for model interpretability.

The goal is not only to predict whether a loan should be approved or rejected, but also to explain the reasons behind each decision, ensuring transparency and fairness in financial decision-making.

## Problem Statement
Banks and financial institutions must evaluate loan applications accurately while maintaining transparency and fairness.  
Traditional models often act as black boxes and fail to explain why a loan is approved or rejected.

This project addresses:
- Accurate loan approval prediction
- Feature-level explainability
- Basic fairness analysis across sensitive attributes

## Solution Approach
- Model: XGBoost (Gradient Boosting Decision Trees)
- Explainability: SHAP (TreeExplainer)
- Fairness Check: Group-wise approval comparison (Gender)
- Deployment: Not included (model evaluation and explainability only)

## Dataset
- Source: Kaggle – Loan Prediction Dataset (Analytics Vidhya)  
- Features include:
  - Applicant Income
  - Coapplicant Income
  - Loan Amount
  - Credit History
  - Education
  - Gender
  - Marital Status
- Target Variable:
  - 1 → Loan Approved  
  - 0 → Loan Rejected  

## Tech Stack
- Python
- scikit-learn
- XGBoost
- SHAP
- Pandas, NumPy
- Matplotlib

## Project Workflow
1. Data loading and preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature encoding  
4. Model training using XGBoost  
5. Model evaluation (Accuracy, Precision, Recall, F1-score)  
6. Explainability using SHAP  
7. Fairness quick-check  

## Model Used: XGBoost
XGBoost is an advanced gradient boosting algorithm that builds trees sequentially, focuses on correcting previous errors, uses regularization to prevent overfitting, and performs exceptionally well on structured/tabular data.

## Explainability with SHAP
SHAP explains model predictions by assigning feature-level contribution values.

SHAP Outputs:
- Global Explanation: Top features influencing loan approval
- Local Explanation: Why a specific loan was approved or rejected

Example Top Features:
1. Credit History  
2. Applicant Income  
3. Loan Amount  
4. Property Area  
5. Education  

## Fairness Quick-Check
A basic fairness analysis is performed by comparing loan approval rates across groups.

Attribute checked:
- Gender

This helps identify potential bias but does not guarantee fairness.

## Reproducibility
- Random seed: 42  
- Train/Test split: 80/20  

## Limitations
- Fairness analysis is preliminary and not causal  
- Dataset is relatively small (~600 records)  
- Model performance depends heavily on Credit History availability  

## Conclusion
This project demonstrates how machine learning can be combined with explainability techniques to build transparent and accountable decision systems for financial applications.

The use of SHAP enables stakeholders to understand model behavior and increases trust in automated loan approval systems.
