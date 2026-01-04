import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

# 1. LOAD MODEL & PREPROCESSORS
model = xgb.XGBClassifier()
model.load_model("xgboost_loan_model.json")

encoders = joblib.load("encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")

print("Model and preprocessing objects loaded successfully\n")

# 2. HELPER FUNCTIONS
def normalize_input(value):
    return value.strip().lower()

value_map = {
    "gender": {"male": "Male", "female": "Female"},
    "married": {"yes": "Yes", "no": "No"},
    "education": {"graduate": "Graduate", "not graduate": "Not Graduate"},
    "self_employed": {"yes": "Yes", "no": "No"},
    "property_area": {
        "urban": "Urban",
        "semiurban": "Semiurban",
        "rural": "Rural"
    }
}

# 3. TAKE USER INPUT
print("Enter Loan Applicant Details:\n")

user_data = {}
user_data["Gender"] = value_map["gender"][normalize_input(input("Gender (Male/Female): "))]
user_data["Married"] = value_map["married"][normalize_input(input("Married (Yes/No): "))]
user_data["Dependents"] = normalize_input(input("Dependents (0 / 1 / 2 / 3+): "))
user_data["Education"] = value_map["education"][normalize_input(input("Education (Graduate/Not Graduate): "))]
user_data["Self_Employed"] = value_map["self_employed"][normalize_input(input("Self Employed (Yes/No): "))]
user_data["ApplicantIncome"] = float(input("Applicant Income: "))
user_data["CoapplicantIncome"] = float(input("Coapplicant Income: "))
user_data["LoanAmount"] = float(input("Loan Amount: "))
user_data["Loan_Amount_Term"] = float(input("Loan Amount Term (in months): "))
user_data["Credit_History"] = float(input("Credit History (1 = Good, 0 = Bad): "))
user_data["Property_Area"] = value_map["property_area"][normalize_input(input("Property Area (Urban/Semiurban/Rural): "))]

# 4. CREATE DATAFRAME
input_df = pd.DataFrame([user_data])

# 5. APPLY ENCODING
categorical_cols = [
    'Gender', 'Married', 'Dependents',
    'Self_Employed', 'Education', 'Property_Area'
]

for col in categorical_cols:
    input_df[col] = encoders[col].transform(input_df[col])

# 6. ENSURE FEATURE ORDER
input_df = input_df[feature_columns]

# 7. PREDICTION
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

# 8. DISPLAY RESULT
print("\nLoan Approval Result")

if prediction == 1:
    print("Loan Status: APPROVED")
else:
    print("Loan Status: REJECTED")

print(f"Approval Probability: {probability:.2f}")

# 9. SHAP WATERFALL EXPLANATION
print("\nGenerating SHAP Waterfall Explanation...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df)

shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=input_df.iloc[0],
        feature_names=input_df.columns
    )
)

plt.show()
