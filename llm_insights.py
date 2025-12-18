from google import genai
import os
import time
from google.genai.errors import ServerError


def generate_insight(churn_prob: float, shap_explanation:dict):
    """
    Generate business-friendly churn explanation using Gemini
    """
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    client = genai.Client(api_key=api_key)
    prompt = f"""
    A customer has a churn probability of {churn_prob:.2f}.

    Top contributing factors:
    {shap_explanation}

    Explain in simple business language:
    1. Why this customer is likely to churn
    2. What actions the business should take to retain the customer
    """

    retries = 3
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text

        except ServerError as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                # Fallback response
                return (
                    "LLM insight is temporarily unavailable due to high load. "
                    "Based on model signals, this customer shows elevated churn risk. "
                    "Recommended actions include proactive outreach, service review, "
                    "and targeted retention offers."
                )