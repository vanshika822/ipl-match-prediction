from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI()

base_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(base_dir, "ipl_model.pkl"))
encoders = joblib.load(os.path.join(base_dir, "label_encoder.pkl"))

class MatchInput(BaseModel):
    team1: str
    team2: str
    venue: str
    temperature: float
    rainfall: float
    toss_winner: str
    toss_decision: str
    season: int
    city: str

@app.get("/")
def home():
    return {"message": "IPL Prediction API Working"}

@app.post("/predict")
def predict(data: MatchInput):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])

    categorical_cols = ['team1', 'team2', 'venue', 'toss_winner', 'toss_decision', 'city']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str).str.strip()
        input_df[col] = encoders[col].transform(input_df[col])

    expected_order = ['team1', 'team2', 'venue', 'temperature', 'rainfall',
                       'toss_winner', 'toss_decision', 'season', 'city']
    input_df = input_df[expected_order]

    prediction = model.predict(input_df)
    winner = encoders['winner'].inverse_transform(prediction)[0]

    return {"predicted_winner": winner}