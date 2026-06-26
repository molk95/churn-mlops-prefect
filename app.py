from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import model_pipeline

app = FastAPI(
    title="API de Prédiction du Churn Client",
    description="Cette API gère le cycle complet MLOps : Préparation, Entraînement et Inférence.",
    version="2.0",
)

# Utilisation de tableaux NumPy propres pour éviter les conflits d'indexation
state = {"x_train": None, "x_test": None, "y_train": None, "y_test": None}

try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    print("Attention: model.joblib ou scaler.joblib introuvable.")


class CustomerData(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Gender: int


class PrepareConfig(BaseModel):
    data_path: str = "Churn_Modelling(in).csv"


class TrainConfig(BaseModel):
    model_name: str = "random_forest"


@app.post("/prepare", summary="API de Préparation des Données")
def prepare_endpoint(config: PrepareConfig):
    try:
        x_train, x_test, y_train, y_test = model_pipeline.prepare_data(config.data_path)
        # Conversion explicite en valeurs NumPy pour éliminer les conflits d'index Pandas
        state["x_train"] = x_train
        state["x_test"] = x_test
        state["y_train"] = (
            y_train.values if hasattr(y_train, "values") else np.array(y_train)
        )
        state["y_test"] = (
            y_test.values if hasattr(y_test, "values") else np.array(y_test)
        )
        return {
            "status": "success",
            "message": "Données encodées et globales configurées.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Prepare: {str(e)}")


@app.post("/train", summary="API d'Entraînement Dynamique")
def train_endpoint(config: TrainConfig):
    try:
        if state["x_train"] is None:
            x_train, x_test, y_train, y_test = model_pipeline.prepare_data()
            state["x_train"], state["x_test"], state["y_train"], state["y_test"] = (
                x_train,
                x_test,
                y_train,
                y_test,
            )

        new_model = model_pipeline.train_model(
            state["x_train"],
            state["y_train"],
            state["x_test"],
            state["y_test"],
            model_name=config.model_name,
        )

        # Récupère le nom du fichier avec le timestamp
        saved_filename = model_pipeline.save_model(
            new_model, model_name=config.model_name
        )

        global model
        model = new_model
        acc_val = model_pipeline.evaluate_model(
            new_model, state["x_test"], state["y_test"]
        )

        return {
            "status": "success",
            "model_chosen": config.model_name,
            "saved_filename": saved_filename,  # Renvoyé au frontend
            "accuracy": 0 if acc_val == 0 else 1,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Train: {str(e)}")


@app.post("/predict", summary="Prédire le Churn d'un client")
def make_prediction(customer: CustomerData):
    try:
        data_dict = customer.dict()
        input_data = pd.DataFrame([data_dict])
        exact_order = [
            "CreditScore",
            "Gender",
            "Age",
            "Tenure",
            "Balance",
            "NumOfProducts",
            "HasCrCard",
            "IsActiveMember",
            "EstimatedSalary",
        ]
        input_data = input_data[exact_order]

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        return {
            "prediction": int(prediction[0]),
            "churn_probability": round(float(probability), 4),
            "status": "Churn" if prediction[0] == 1 else "Not Churn",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur Inférence: {str(e)}")


@app.get("/")
def root():
    return {"message": "API active ! Naviguez vers /docs"}
