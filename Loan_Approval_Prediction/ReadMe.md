# Loan Approval Prediction using XGBoost & SHAP

## Project Overview
This project builds an explainable loan approval prediction system using XGBoost for classification and SHAP (SHapley Additive exPlanations) for model interpretability.

The system predicts whether a loan should be approved or rejected and explains why the decision was made at both global and individual applicant levels. A basic fairness check across gender is also included.

---

## Problem Statement
Banks and financial institutions need accurate, transparent, and fair loan approval systems.

Traditional ML models often act as black boxes and cannot explain their decisions.

This project addresses:
- Accurate loan approval prediction  
- Feature-level explainability for trust and transparency  
- A simple fairness diagnostic across sensitive attributes  

---

## Dataset
- Source: Kaggle – Loan Prediction Dataset (Analytics Vidhya)  
- File used: `train_u6lujuX_CVtuZ9i.csv`  
- Size: ~600 records  

### Input Features
- Gender  
- Married  
- Dependents  
- Education  
- Self Employed  
- Applicant Income  
- Coapplicant Income  
- Loan Amount  
- Loan Amount Term  
- Credit History  
- Property Area  

### Target Variable
- Loan_Status = 1 → Approved  
- Loan_Status = 0 → Rejected  

---

## Tech Stack
- Python  
- Pandas, NumPy  
- scikit-learn  
- XGBoost  
- SHAP  
- Matplotlib  
- Joblib  

---

## Project Workflow

### 1. Data Preprocessing
- Missing categorical values filled with mode  
- Missing numerical values filled with median  
- Categorical features encoded using LabelEncoder  
- Feature order saved for consistent testing  

### 2. Model Training
- Algorithm: XGBoost Classifier  
- Hyperparameters:
  - n_estimators = 200
  - max_depth = 4
  - learning_rate = 0.05
  - subsample = 0.8
  - colsample_bytree = 0.8
  - random_state = 42
- Train/test split: 80/20  

### 3. Model Evaluation
Metrics used:
- Accuracy  
- Precision, Recall, F1-score (classification report)

---

## Explainability with SHAP

### Global Explainability
- SHAP summary plot shows most influential features.
- Top features typically:
  1. Credit History  
  2. Applicant Income  
  3. Loan Amount  
  4. Property Area  
  5. Education  

### Local Explainability
- SHAP waterfall plot explains individual predictions.
- Shows how each feature pushes the decision toward approval or rejection.

---

## Fairness Quick-Check
A basic group fairness analysis is performed:
- Approval rates are compared across Gender groups.
- This is a diagnostic check only and does not guarantee fairness.

---

## Prediction (Testing Phase)
The trained model:
- Takes user input via console  
- Encodes inputs using saved encoders  
- Ensures feature order consistency  
- Predicts approval status  
- Shows approval probability  
- Generates SHAP explanation for that individual case  

---

## Reproducibility
- Random seed: 42  
- Train/test split: 80/20  
- Feature order stored in feature_columns.pkl  
- Encoders stored in encoders.pkl  

---

## Limitations
- Dataset is small (~600 records)  
- Fairness analysis is only a surface-level check  
- Credit history strongly dominates predictions  
- No deployment layer (CLI only)

---

## Conclusion
This project demonstrates how machine learning models can be made transparent using SHAP while maintaining strong predictive performance with XGBoost.

It provides:
- Accurate loan approval predictions  
- Human-understandable explanations  
- A simple fairness diagnostic  

This combination makes the system more trustworthy and suitable for responsible AI in financial applications.
