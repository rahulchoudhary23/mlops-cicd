from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("app/model.pkl")


@app.get("/")
def health_check():
    return {"status": "Model is running"}


@app.get("/info")
def model_info():
    return {"model": "Logistic Regression", "dataset": "Iris"}


@app.post("/predict")
def predict(data: dict):

    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)

    return {"prediction": int(prediction[0])}
