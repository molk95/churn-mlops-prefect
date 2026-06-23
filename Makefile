# Variables
PYTHON = venv/bin/python
PIP = venv/bin/pip
UVICORN = venv/bin/uvicorn
STREAMLIT = venv/bin/streamlit
PYTEST = venv/bin/pytest

.PHONY: install format lint security test run-api run-ui prefect-all

# 1. Installation des dépendances
install:
	$(PIP) install -r requirements.txt

# 2. Formatage et Qualité du code
format:
	$(PYTHON) -m black pipeline_prefect.py model_pipeline.py app.py app_streamlit.py

lint:
	$(PYTHON) -m flake8 model_pipeline.py app.py app_streamlit.py
	$(PYTHON) -m pylint model_pipeline.py app.py app_streamlit.py --disable=C,R

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