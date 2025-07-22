from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
import os, io, joblib, numpy as np, pandas as pd, shap, requests, openai
from reportlab.pdfgen import canvas

app = FastAPI()

# Load model
import pathlib
MODEL_PATH = pathlib.Path("ml/model.pkl")
model = joblib.load(MODEL_PATH)



# API key
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != os.getenv("API_KEY", "yoursecretkey123"):
        raise HTTPException(status_code=401, detail="Invalid API Key")

# Input schema
class ApplicantData(BaseModel):
    income: float
    credit_score: float
    employment_length: float
    loan_amount: float
    dti: float
    self_employed: float

@app.get("/")
def home():
    return {"msg": "Smart Mortgage Assistant is live!"}

@app.post("/predict")
def predict(data: ApplicantData, auth=Depends(verify_api_key)):
    X = np.array([[v for v in data.dict().values()]])
    pred = model.predict_proba(X)[0][1]
    recommendation = "Approve" if pred > 0.7 else "Manual Review" if pred > 0.4 else "Reject"
    record = data.dict()
    record["risk_score"] = round(float(pred), 2)
    record["recommendation"] = recommendation
    record["timestamp"] = datetime.utcnow().isoformat()
    log_to_airtable(record)
    return record

@app.post("/explain")
def explain(data: ApplicantData, auth=Depends(verify_api_key)):
    X_df = pd.DataFrame([data.dict()])
    explainer = shap.Explainer(model)
    shap_values = explainer(X_df)
    return dict(zip(X_df.columns, shap_values.values[0]))

@app.post("/explain/llm")
def explain_llm(data: ApplicantData, auth=Depends(verify_api_key)):
    X_df = pd.DataFrame([data.dict()])
    explainer = shap.Explainer(model)
    shap_values = explainer(X_df)
    contrib = dict(zip(X_df.columns, shap_values.values[0]))

    pred = model.predict_proba(X_df)[0][1]
    recommendation = "Approve" if pred > 0.7 else "Manual Review" if pred > 0.4 else "Reject"

    prompt = f"""A mortgage applicant submitted: {X_df.to_dict(orient='records')[0]}.
Risk score: {round(pred, 2)}. Recommendation: {recommendation}.
SHAP values: {contrib}.
Explain why this result happened, in plain English, and how to improve it."""

    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
        messages=[
            {"role": "system", "content": "You are an expert mortgage advisor."},
            {"role": "user", "content": prompt}
        ]
    )
    return {"explanation": response['choices'][0]['message']['content'].strip()}

@app.post("/export/pdf")
def export_pdf(data: ApplicantData, auth=Depends(verify_api_key)):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"exports/report_{timestamp}.pdf"
    c = canvas.Canvas(filename)
    y = 750
    c.drawString(50, y, "Smart Mortgage Assistant Report")
    y -= 30
    for field, value in data.dict().items():
        c.drawString(50, y, f"{field}: {value}")
        y -= 20
    pred = model.predict_proba([[v for v in data.dict().values()]])[0][1]
    rec = "Approve" if pred > 0.7 else "Manual Review" if pred > 0.4 else "Reject"
    c.drawString(50, y - 20, f"Risk Score: {round(pred, 2)}")
    c.drawString(50, y - 40, f"Recommendation: {rec}")
    c.save()
    return FileResponse(filename, media_type='application/pdf', filename="mortgage_report.pdf")

@app.post("/export/excel")
def export_excel(data: ApplicantData, auth=Depends(verify_api_key)):
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"exports/report_{timestamp}.xlsx"
    pred = model.predict_proba([[v for v in data.dict().values()]])[0][1]
    rec = "Approve" if pred > 0.7 else "Manual Review" if pred > 0.4 else "Reject"
    df = pd.DataFrame([data.dict()])
    df["risk_score"] = round(pred, 2)
    df["recommendation"] = rec
    df.to_excel(filename, index=False)
    return FileResponse(filename, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename="mortgage_report.xlsx")

def log_to_airtable(record):
    url = f"https://api.airtable.com/v0/{os.getenv('AIRTABLE_BASE_ID')}/{os.getenv('AIRTABLE_TABLE', 'Predictions')}"
    headers = {
        "Authorization": f"Bearer {os.getenv('AIRTABLE_API_KEY')}",
        "Content-Type": "application/json"
    }
    requests.post(url, headers=headers, json={"fields": record})
