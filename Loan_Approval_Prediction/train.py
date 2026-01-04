# IMPORTS
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
# 1. LOAD DATASET
df = pd.read_csv("Dataset/train_u6lujuX_CVtuZ9i.csv")

# 2. HANDLE MISSING VALUES (NO inplace WARNING)
categorical_cols = [
    'Gender', 'Married', 'Dependents',
    'Self_Employed', 'Education', 'Property_Area'
]

numerical_cols = [
    'LoanAmount', 'Loan_Amount_Term', 'Credit_History'
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
# 3. ENCODE CATEGORICAL FEATURES
encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le   # store encoder for testing
# 4. SPLIT FEATURES & TARGET
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Save feature order (VERY IMPORTANT for testing)
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 5. TRAIN XGBOOST MODEL
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# 6. MODEL EVALUATION
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. SAVE TRAINED OBJECTS (FOR TESTING)
model.save_model("xgboost_loan_model.json")
joblib.dump(encoders, "encoders.pkl")

# (Optional – for debugging / learning)
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)

print("\n Model and preprocessing objects saved successfully")
# 8. SHAP EXPLAINABILITY
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global explanation
shap.summary_plot(shap_values, X_test)

# Top 5 important features
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": shap_importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 5 Important Features:")
print(feature_importance.head(5))

# Local explanation (one sample)
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns
    )
)

# 9. FAIRNESS QUICK CHECK (Gender)
df_fair = X_test.copy()
df_fair["Loan_Status"] = y_test.values
df_fair["Gender"] = df.loc[X_test.index, "Gender"]

approval_rate = df_fair.groupby("Gender")["Loan_Status"].mean()
print("\nApproval Rate by Gender:")
print(approval_rate)
