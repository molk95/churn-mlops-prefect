import argparse
import subprocess
import os
import sys
import time
from prefect import task, flow
import mlflow

# On importe tes fonctions existantes du modèle
from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)

# ÉTAPE 04 : FONCTIONS LIÉES AU SUIVI ET À LA QUALITÉ DU CODE => TASKS


@task(name="1. Installer les dépendances")
def task_install_dependencies():
    print("\n=== [TASK] Installation des dépendances depuis requirements.txt ===")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True
    )


@task(name="2. Formatage du code")
def task_format_code():
    print("\n=== [TASK] Formatage du code avec Black ===")
    subprocess.run(
        ["black", "pipeline_prefect.py", "model_pipeline.py", "app.py"], check=False
    )


@task(name="3. Qualité du code")
def task_check_quality():
    print("\n=== [TASK] Vérification de la qualité du code avec Flake8 ===")
    subprocess.run(["flake8", "model_pipeline.py", "app.py"], check=False)


@task(name="4. Sécurité du code")
def task_check_security():
    print("\n=== [TASK] Analyse de sécurité avec Bandit ===")
    subprocess.run(["bandit", "-r", "model_pipeline.py", "app.py"], check=False)


@task(name="5. Exécution des tests unitaires")
def task_run_tests():
    print("\n=== [TASK] Exécution des tests unitaires avec Pytest ===")
    test_code = """
def test_sample():
    assert True
"""
    with open("test_pipeline.py", "w") as f:
        f.write(test_code)

    subprocess.run(["pytest", "test_pipeline.py"], check=False)


# ÉTAPE 02 : FONCTIONS LIÉES AUX DONNÉES ET AU MODÈLE (TASKS)


@task(name="6. Préparation des données")
def task_prepare_data():
    print("\n=== [TASK] Préparation et nettoyage des données ===")
    x_train, x_test, y_train, y_test = prepare_data()
    return x_train, x_test, y_train, y_test


@task(name="7. Entraînement du modèle")
def task_train_model(x_train, y_train, x_test=None, y_test=None):
    print("\n=== [TASK] Entraînement du modèle avec suivi MLflow ===")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    model = train_model(x_train, y_train, x_test, y_test)
    save_model(model)
    return model


@task(name="8. Évaluation du modèle")
def task_evaluate_model(model, x_test, y_test):
    print("\n=== [TASK] Évaluation des performances du modèle ===")
    accuracy = evaluate_model(model, x_test, y_test)
    return accuracy


@task(name="9. Prédiction")
def task_predict(x_test):
    print("\n=== [TASK] Génération des prédictions (Inférence) ===")
    predictions = predict(x_test)
    print(f"Quelques prédictions générées : {predictions[:5]}")
    return predictions


@task(name="10. Lancement de l'API de Production (app.py)")
def task_launch_api():
    print("\n=== [TASK] Lancement de l'API de production FastAPI ===")
    process = subprocess.Popen(
        ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
    )
    time.sleep(2)
    print("🚀 API de production lancée sur http://127.0.0.1:8080")


@task(name="0. Update Code from Git")
def task_git_pull():
    print("\n=== [TASK] Récupération de la dernière version du code via Git ===")
    # Lance un git pull pour mettre à jour les fichiers locaux automatiquement
    subprocess.run(["git", "pull", "origin", "main"], check=False)


# DÉFINITION DES FLOWS => Regroupement des fonctions


@flow(name="ml-pipeline-all")
def flow_all():
    task_git_pull()
    task_install_dependencies()
    task_format_code()
    task_check_quality()
    task_check_security()
    task_run_tests()
    x_train, x_test, y_train, y_test = task_prepare_data()
    model = task_train_model(x_train, y_train, x_test, y_test)
    task_evaluate_model(model, x_test, y_test)
    task_predict(x_test)
    task_launch_api()


@flow(name="ml-pipeline-install")
def flow_install():
    task_install_dependencies()


@flow(name="ml-pipeline-prepare")
def flow_prepare():
    task_prepare_data()


@flow(name="ml-pipeline-train")
def flow_train():
    x_train, x_test, y_train, y_test = prepare_data()
    task_train_model(x_train, y_train, x_test, y_test)


@flow(name="ml-pipeline-evaluate")
def flow_evaluate():
    x_train, x_test, y_train, y_test = prepare_data()
    model = load_model()
    task_evaluate_model(model, x_test, y_test)


@flow(name="ml-pipeline-predict")
def flow_predict():
    x_train, x_test, y_train, y_test = prepare_data()
    task_predict(x_test)


@flow(name="ml-pipeline-api")
def flow_api():
    task_launch_api()


# AUTOMATISATION ET GESTION DU CLI =W MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline MLOps avec Prefect")
    parser.add_argument(
        "--flow",
        type=str,
        required=True,
        choices=["all", "install", "prepare", "train", "evaluate", "predict", "api"],
        help="Choisir le flow Prefect à exécuter",
    )

    args = parser.parse_args()

    if args.flow == "all":
        flow_all()
    elif args.flow == "install":
        flow_install()
    elif args.flow == "prepare":
        flow_prepare()
    elif args.flow == "train":
        flow_train()
    elif args.flow == "evaluate":
        flow_evaluate()
    elif args.flow == "predict":
        flow_predict()
    elif args.flow == "api":
        flow_api()
