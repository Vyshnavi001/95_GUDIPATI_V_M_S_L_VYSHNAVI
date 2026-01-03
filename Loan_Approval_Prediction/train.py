# pip install SHAP
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset/train_u6lujuX_CVtuZ9i.csv")
df.head()
df.info()
df.isnull().sum()
categorical_cols = ['Gender', 'Married', 'Dependents',
                    'Self_Employed', 'Education', 'Property_Area']
numerical_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])
X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
y = df['Loan_Status'].map({'Y': 1, 'N': 0})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap_importance = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': shap_importance
}).sort_values(by='Importance', ascending=False)

print(feature_importance.head(5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=X_test.iloc[0],
        feature_names=X_test.columns
    )
)
df_test = X_test.copy()
df_test['Loan_Status'] = y_test.values
df_test['Gender'] = df.loc[X_test.index, 'Gender']

approval_rate = df_test.groupby('Gender')['Loan_Status'].mean()
print(approval_rate)