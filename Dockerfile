FROM python:3.8.6-buster

COPY api /api
COPY alien /alien
COPY model.joblib /model.joblib
COPY requirements.txt /requirements.txt
COPY data-science-313109-c3a34aab7075.json /credentials.json

RUN pip install -r requirements.txt

CMD uvicorn api.alien_api:api --host 0.0.0.0 --port $PORT