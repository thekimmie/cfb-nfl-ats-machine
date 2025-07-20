# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os

app = FastAPI()

# Enable CORS so Retool can access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # (or restrict to Retool domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data from CSV or prepare JSON
DATA_PATH = os.path.join(os.path.dirname(__file__), "sample_data.csv")
data_df = pd.read_csv(DATA_PATH)

@app.get("/api/model-data")
def get_model_data():
    return data_df.to_dict(orient="records")

# Render looks for this
handler = app
