import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # Modifié
from sklearn.linear_model import LogisticRegression # Modifié
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import datetime

def prepare_data(data_path="Churn_Modelling(in).csv"):
    # importation du encodeur
    encocoder = LabelEncoder()
    # chargement data du fichier csv
    data = pd.read_csv(data_path)
    # supprimer les colonnes inutiles pays et nom
    data = data.drop(["Surname", "Geography"], axis=1)
    # encodatages colonnes gender
    data["Gender"] = encocoder.fit_transform(data["Gender"])
    # supprimer les lignes des donnees manquantes
    data = data.dropna()
    # separation des donnees en X characterisques et class y
    # supprimer les colones initules pour X
    X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = data["Exited"]
    # 80% train , 20% test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    # normalisation
    scaler = StandardScaler()
    # transformation des data d'entrainement
    x_train_scaled = scaler.fit_transform(x_train)
    # transformation des data de test
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, "scaler.joblib")
    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train, y_train, model_name="random_forest"):
    if model_name == "logistic_regression":
        model = LogisticRegression(random_state=1)
    elif model_name == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=1)
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=1)
    # training sur les donnnées x_train et les classes y_train    
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    y_prd = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_prd)
    # afficher precision du modèle
    print(f"Accuracy: {accuracy}")
    return accuracy


def predict(features):
    model = joblib.load("model.joblib")
    prediction = model.predict(features)
    return prediction


def save_model(model, model_name="model"):
    # Génère un nom du style: random_forest_20260623_2340.joblib
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{model_name}_{timestamp}.joblib"
    
    joblib.dump(model, filename)
    # On sauvegarde aussi une copie fixe 'model.joblib' pour que l'API /predict fonctionne sans interruption
    joblib.dump(model, "model.joblib") 
    
    print(f"Model saved to {filename} and model.joblib")
    return filename


def load_model(model_path="model.joblib"):
    # model
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model
