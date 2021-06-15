import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@api.get("/")
def index():
    return {"greeting": "Welcome to alien sightings api"}

@api.get("/predict")
def predict(state, season):

    data = pd.read_csv('/Users/juan/code/Polanket/alien/raw_data/final_df.csv')

    print(data.head(1))

    X = data.loc[np.logical_and(data['state'] == state, data['season_y'] == season)]

    print(X.tail(1))

    # ⚠️ TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(prediction=pred)