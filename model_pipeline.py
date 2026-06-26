import datetime
import os
import time
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import psutil  # Pour les métriques système
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_data(data_path="Churn_Modelling(in).csv"):
    encoder = LabelEncoder()
    data = pd.read_csv(data_path)
    data = data.drop(["Surname", "Geography"], axis=1)
    data["Gender"] = encoder.fit_transform(data["Gender"])
    data = data.dropna()

    X = data.drop(["Exited", "RowNumber", "CustomerId"], axis=1)
    y = data["Exited"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, "scaler.joblib")
    return x_train_scaled, x_test_scaled, y_train, y_test


def train_model(x_train, y_train, x_test=None, y_test=None, model_name="random_forest"):
    """Entraîne un modèle et harmonise le logging pour optimiser les graphiques

    comparatifs (Scatter, Parallel Coordinates) dans l'UI MLflow.
    """
    print(f"\n=== Début de l'entraînement : {model_name} ===")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("Churn_Analytics_Experiment")
    mlflow.sklearn.autolog()
    mlflow.enable_system_metrics_logging()

    # Initialisation des modèles et alignement des hyperparamètres pour l'UI
    if model_name == "random_forest":
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestClassifier(**params)
        # Hyperparamètres spécifiques pour l'alignement visuel
        hparams_to_log = {
            "complexity_param": params["max_depth"],
            "regularization_c": 0.0,
        }

    elif model_name == "logistic_regression":
        params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
        model = LogisticRegression(**params)
        hparams_to_log = {"complexity_param": 0, "regularization_c": params["C"]}

    elif model_name == "gradient_boosting":
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "random_state": 42,
        }
        model = GradientBoostingClassifier(**params)
        hparams_to_log = {
            "complexity_param": params["max_depth"],
            "regularization_c": 0.0,
        }
    else:
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        model = RandomForestClassifier(**params)
        model_name = "random_forest"
        hparams_to_log = {
            "complexity_param": params["max_depth"],
            "regularization_c": 0.0,
        }

    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024 * 1024)

    with mlflow.start_run(run_name=f"Run_{model_name}"):
        # Log des paramètres alignés pour faciliter les tracés de contour et coordonnées parallèles
        mlflow.log_param("chosen_algorithm", model_name)
        for k, v in hparams_to_log.items():
            mlflow.log_param(k, v)

        model.fit(x_train, y_train)

        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)
        cpu_usage = psutil.cpu_percent(interval=None)

        mlflow.log_metric("sys_training_duration_sec", end_time - start_time)
        mlflow.log_metric("sys_cpu_usage_percent", cpu_usage)
        mlflow.log_metric("sys_memory_delta_mb", end_memory - start_memory)

        if x_test is not None and y_test is not None:
            predictions = model.predict(x_test)
            acc = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, zero_division=0)
            recall = recall_score(y_test, predictions, zero_division=0)
            f1 = f1_score(y_test, predictions, zero_division=0)

            # Métriques explicites pour tes graphiques
            mlflow.log_metric("model_accuracy", float(acc))
            mlflow.log_metric("model_precision", float(precision))
            mlflow.log_metric("model_recall", float(recall))
            mlflow.log_metric("model_f1_score", float(f1))

            print(f"Metrics - Accuracy: {acc*100:.2f}%")

        # Explicitly log the model and scaler as artifacts
        mlflow.sklearn.log_model(model, "model")
        if os.path.exists("scaler.joblib"):
            mlflow.log_artifact("scaler.joblib")

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
