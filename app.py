from fastapi import FastAPI
from predict import predict_churn
from explain import explain_prediction
from llm_insights import generate_insight
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn with ML + SHAP + LLM insights",
    version="1.0"
)
class CustomerInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    PaymentMethod: str
    InternetService: str
    TechSupport: str

@app.post("/predict")
def predict(customer: CustomerInput):
    input_data = customer.model_dump()
    churn_prob = predict_churn(input_data)
    shap_exp = explain_prediction(input_data)
    insight = generate_insight(churn_prob, shap_exp)
    return {    
        "churn_probability": churn_prob,
        "top_factors": shap_exp,
        "llm_insight": insight
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )