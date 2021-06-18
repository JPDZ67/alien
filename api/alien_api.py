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

BUCKET_NAME = 'ufo_sightings'
BUCKET_TRAIN_DATA_PATH = 'data/final_df.csv'


@api.get("/")
def index():
    return {"greeting": "Welcome to alien sightings api"}

@api.get("/predict")
def predict(state, season):

    data = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")

    print(data.head(1))

    X = data.loc[np.logical_and(data['state'] == state, data['season_y'] == season)]

    X.dropna(inplace=True)
    X.drop(columns=['year_season'],
           inplace=True,
           errors='ignore')

    X = X.drop('sightings_t+1', axis=1)

    X = X.tail(1)

    # ⚠️ TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(prediction=pred)

