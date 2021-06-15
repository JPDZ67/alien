import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

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
def predict(state,
            year_season,
            avg_duration,
            sightings_cities,
            shape,
            sightings_days,
            year,
            season,
            population):

    X = pd.DataFrame(dict(
        state=[state],
        year_season=[year_season],
        avg_duration=[float(avg_duration)],
        sightings_cities=[int(sightings_cities)],
        shape=[shape],
        sightings_days=[int(sightings_days)],
        year=[int(year)],
        season=[season],
        population=[int(population)]))

    # ⚠️ TODO: get model from GCP

    # pipeline = get_model_from_gcp()
    pipeline = joblib.load('model.joblib')

    # make prediction
    results = pipeline.predict(X)

    # convert response from numpy to python type
    pred = float(results[0])

    return dict(prediction=pred)