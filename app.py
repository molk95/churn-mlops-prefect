from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialiser l'application FastAPI
app = FastAPI(
    title="API de Prédiction du Churn Client",
    description="Cette API utilise un modèle Random Forest pour prédire si un client va quitter la banque.",
    version="1.0"
)

# Charger le modèle et le scaler sauvegardés par le pipeline
try:
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    print("Attention: model.joblib ou scaler.joblib introuvable. Lance ton pipeline train d'abord.")

# Définir la structure des données attendues (le format du client)
# Remplace ces champs par les vraies caractéristiques (features) de ton dataset Churn_Modelling
class CustomerData(BaseModel):
    CreditScore: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Gender: int  # Encodé ( 0 F, 1 H)

# Définir la route HTTP POST pour les prédictions
@app.post("/predict", summary="Prédire le Churn d'un client")
def make_prediction(customer: CustomerData):
    try:
        # Convertir les données reçues en dictionnaire
        data_dict = customer.dict()
        
        # Créer le DataFrame
        input_data = pd.DataFrame([data_dict])
        
        # FORCE L'ORDRE EXACT DES COLONNES
        # On réordonne les colonnes du DataFrame pour correspondre au fit d'origine
        exact_order = [
            "CreditScore", "Gender", "Age", "Tenure", 
            "Balance", "NumOfProducts", "HasCrCard", 
            "IsActiveMember", "EstimatedSalary"
        ]
        input_data = input_data[exact_order]
        
        # Appliquer le scaler et prédire
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]
        
        return {
            "prediction": int(prediction[0]),
            "churn_probability": round(float(probability), 4),
            "status": "Quitte la banque" if prediction[0] == 1 else "Reste dans la banque"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction: {str(e)}")

# Petite route de base pour vérifier que l'API est en vie
@app.get("/")
def root():
    return {"message": "API de prédiction active et prête ! Accède à /docs pour la tester."}