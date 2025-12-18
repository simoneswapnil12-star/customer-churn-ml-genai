import joblib
import pandas as pd

model = joblib.load(r"E:\Coding Projects\GENENV\genenv\customer_churn\customer_churn_model.pkl")
features = joblib.load(r"E:\Coding Projects\GENENV\genenv\customer_churn\customer_churn_features.pkl")

def predict_churn(input_data: dict):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=features, fill_value=0)

    prob = model.predict_proba(df)[0][1]
    return prob