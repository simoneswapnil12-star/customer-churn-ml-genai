import pandas as pd
import joblib
from sklearn.model_selection import train_test_split   
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

df = pd.read_csv(r"E:\Coding Projects\GENENV\genenv\customer_churn\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.columns = df.columns.str.strip()
df.drop(columns=['customerID'], inplace=True)


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges'])

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = pd.get_dummies(df, drop_first=True)

X=df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),   
    ('model', LogisticRegression(max_iter=1000))])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict_proba(X_test)[:, 1]

print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

joblib.dump(pipeline, r"E:\Coding Projects\GENENV\genenv\customer_churn\customer_churn_model.pkl")
joblib.dump(X.columns, r"E:\Coding Projects\GENENV\genenv\customer_churn\customer_churn_features.pkl")