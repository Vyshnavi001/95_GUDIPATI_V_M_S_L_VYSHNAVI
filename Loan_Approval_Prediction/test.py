import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# 1. LOAD SAVED MODEL & OBJECTS
model = xgb.XGBClassifier()
model.load_model("xgboost_loan_model.json")

encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

print("Model and preprocessing objects loaded successfully\n")

# 2. TAKE USER INPUT
print("Enter Loan Applicant Details:\n")

user_data = {}

user_data["Gender"] = input("Gender (Male/Female): ")
user_data["Married"] = input("Married (Yes/No): ")
user_data["Dependents"] = input("Dependents (0 / 1 / 2 / 3+): ")
user_data["Education"] = input("Education (Graduate/Not Graduate): ")
user_data["Self_Employed"] = input("Self Employed (Yes/No): ")
user_data["ApplicantIncome"] = float(input("Applicant Income: "))
user_data["CoapplicantIncome"] = float(input("Coapplicant Income: "))
user_data["LoanAmount"] = float(input("Loan Amount: "))
user_data["Loan_Amount_Term"] = float(input("Loan Amount Term (in months): "))
user_data["Credit_History"] = float(input("Credit History (1 = Good, 0 = Bad): "))
user_data["Property_Area"] = input("Property Area (Urban/Semiurban/Rural): ")

# 3. CONVERT INPUT TO DATAFRAME
input_df = pd.DataFrame([user_data])
# 4. APPLY ENCODING
categorical_cols = [
    'Gender', 'Married', 'Dependents',
    'Self_Employed', 'Education', 'Property_Area'
]

for col in categorical_cols:
    input_df[col] = encoders[col].transform(input_df[col])
# 5. ENSURE FEATURE ORDER
input_df = input_df[feature_columns]
# 6. MAKE PREDICTION
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]
# 7. DISPLAY RESULT
print("\n Loan Approval Result")
print("----------------------")

if prediction == 1:
    print(" Loan Status: APPROVED")
else:
    print(" Loan Status: REJECTED")

print(f" Approval Probability: {probability:.2f}")
