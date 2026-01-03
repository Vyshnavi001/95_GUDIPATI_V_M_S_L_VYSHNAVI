import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Load trained model
# -----------------------------
model = xgb.XGBClassifier()
model.load_model("xgboost_loan_model.json")

# -----------------------------
# 2. Load test dataset
# -----------------------------
test_df = pd.read_csv("Dataset/test_Y3wMUE5_7gLdaTN.csv")

print("Test Data Loaded Successfully")
print(test_df.head())

# -----------------------------
# 3. Handle missing values
# -----------------------------
categorical_cols = ['Gender', 'Married', 'Dependents',
                    'Self_Employed', 'Education', 'Property_Area']

numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']

for col in categorical_cols:
    test_df[col].fillna(test_df[col].mode()[0], inplace=True)

for col in numerical_cols:
    test_df[col].fillna(test_df[col].median(), inplace=True)

# -----------------------------
# 4. Encode categorical features
# -----------------------------
le = LabelEncoder()
for col in categorical_cols:
    test_df[col] = le.fit_transform(test_df[col])

# -----------------------------
# 5. Prepare test features
# -----------------------------
X_test = test_df.drop('Loan_ID', axis=1)

# -----------------------------
# 6. Make predictions
# -----------------------------
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Convert numeric output to labels
test_df['Loan_Status_Predicted'] = np.where(predictions == 1, 'Approved', 'Rejected')
test_df['Approval_Probability'] = probabilities

# -----------------------------
# 7. Display results
# -----------------------------
print("\nPrediction Results:")
print(test_df[['Loan_ID', 'Loan_Status_Predicted', 'Approval_Probability']].head())

# -----------------------------
# 8. Save predictions
# -----------------------------
test_df.to_csv("loan_test_predictions.csv", index=False)
print("\nPredictions saved to loan_test_predictions.csv")
