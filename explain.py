import shap
import joblib
import pandas as pd

model = joblib.load(r"E:\Coding Projects\GENENV\genenv\customer_churn\customer_churn_model.pkl")
features = joblib.load(r"E:\Coding Projects\GENENV\genenv\customer_churn\customer_churn_features.pkl")  

background = pd.DataFrame([[0]* len(features)], columns=features)
masker = shap.maskers.Independent(background)

explainer = shap.LinearExplainer(model.named_steps['model'], masker)

def explain_prediction(input_data):
    df = pd.DataFrame([input_data])
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=features, fill_value=0)

    shap_values = explainer(df)

    # Pair feature names with their SHAP values, sort by absolute importance, take top 5
    explanation= dict(sorted(zip(features, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True)[:5])
    

    return explanation