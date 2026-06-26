# Variables
PYTHON = venv/bin/python
PIP = venv/bin/pip
UVICORN = venv/bin/uvicorn
STREAMLIT = venv/bin/streamlit
PYTEST = venv/bin/pytest
PREFECT = venv/bin/prefect

# Docker Variables
DOCKER_API_NAME = molk_saouabi_sde_2_mlops_api
DOCKER_UI_NAME = molk_saouabi_sde_2_mlops_ui
DOCKER_HUB_USER = molksaouabi
DOCKER_API_TAG = $(DOCKER_HUB_USER)/$(DOCKER_API_NAME):latest
DOCKER_UI_TAG = $(DOCKER_HUB_USER)/$(DOCKER_UI_NAME):latest

.PHONY: install format lint security test run-api run-ui prefect-all run-prefect docker-build docker-tag docker-push docker-run compose-up compose-down

# 1. Installation des dépendances
install:
	$(PIP) install -r requirements.txt

# 2. Formatage et Qualité du code
format:
	$(PYTHON) -m black pipeline_prefect.py model_pipeline.py app.py app_streamlit.py

lint:
	$(PYTHON) -m flake8 model_pipeline.py app.py app_streamlit.py
	$(PYTHON) -m pylint model_pipeline.py app.py app_streamlit.py --disable=C,R,W

security:
	$(PYTHON) -m bandit -r model_pipeline.py app.py

# 3. Tests unitaires
test:
	$(PYTEST) test_main.py test_pipeline.py -v

# 4. Lancement de l'API Backend (FastAPI)
run-api:
	$(UVICORN) app:app --host 0.0.0.0 --port 8080 --reload

# 5. Lancement de l'Interface Frontend (Streamlit)
run-ui:
	$(STREAMLIT) run app_streamlit.py --server.port 8501

# 6. Exécution du pipeline complet via le CLI Prefect
prefect-all:
	$(PYTHON) pipeline_prefect.py --flow all

# Lancement du serveur local Prefect
run-prefect:
	$(PREFECT) server start

# Lance l'interface MLflow avec le backend SQLite en arrière-plan
run-mlflow:
	@echo "🚀 Lancement de MLflow avec backend SQLite sur http://127.0.0.1:5000"
	@mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000 &

# Nettoie les artefacts locaux et la DB si tu veux réinitialiser tes expériences
clean-mlflow:
	@rm -rf mlruns mlflow.db
	@echo "🧹 Historique MLflow réinitialisé."

# 7. Tâches Docker
docker-build:
	docker build -t $(DOCKER_API_NAME) -f Dockerfile .
	docker build -t $(DOCKER_UI_NAME) -f Dockerfile.ui .

docker-tag:
	docker tag $(DOCKER_API_NAME) $(DOCKER_API_TAG)
	docker tag $(DOCKER_UI_NAME) $(DOCKER_UI_TAG)

docker-push: docker-tag
	docker push $(DOCKER_API_TAG)
	docker push $(DOCKER_UI_TAG)

docker-run:
	docker run -d -p 8080:8080 --name churn-prediction-api $(DOCKER_API_NAME)
	docker run -d -p 8501:8501 --name churn-prediction-ui $(DOCKER_UI_NAME)

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down