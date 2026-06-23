# Churn Analytics MLOps Pipeline

This project is an end-to-end MLOps pipeline for predicting customer churn. It includes data preparation, model training orchestration, a REST API for inference, and an interactive Streamlit dashboard for monitoring and pipeline administration.

## Features

- **Data Pipeline**: Automated data preparation and feature engineering.
- **Model Training**: Support for multiple models (Random Forest, Logistic Regression, Gradient Boosting) via dynamic endpoints.
- **REST API**: Built with FastAPI to handle preparation, training, and prediction requests.
- **Interactive Dashboard**: A Streamlit UI to visualize data, test predictions, and trigger training jobs.
- **Orchestration**: Prefect integration for robust pipeline execution.

## Project Structure

- `app.py`: FastAPI backend application.
- `app_streamlit.py`: Streamlit frontend application.
- `model_pipeline.py`: Core machine learning logic (data prep, training, evaluation).
- `pipeline_prefect.py` / `deploiement_prefect.py`: Prefect workflow orchestration.
- `Makefile`: Commands for running, testing, formatting, and linting the application.

## Prerequisites

- Python 3.8+
- `pip` package manager

## Installation

1. **Clone the repository** and navigate to the project directory (if not already there):
   ```bash
   cd ml_project
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   make install
   # Or manually: pip install -r requirements.txt
   ```

## How to Run

You can run the different components of the project using the provided `Makefile`:

### 1. Start the Backend API (FastAPI)
The backend provides the endpoints for inference, preparation, and training.
```bash
make run-api
```
*The API will be available at `http://localhost:8080` (Swagger docs at `http://localhost:8080/docs`).*

### 2. Start the Frontend Dashboard (Streamlit)
The frontend provides a UI to interact with the API, visualize data, and manage models.
```bash
make run-ui
```
*The dashboard will be available at `http://localhost:8501`.*

### 3. Run the Full Prefect Pipeline
To execute the complete MLOps pipeline using Prefect (preparation, training, evaluation):
```bash
make prefect-all
```

## Testing & Quality

The project includes unit tests, formatting, and linting tools to maintain code quality.

- **Run unit tests**:
  ```bash
  make test
  ```

- **Format code** (Black):
  ```bash
  make format
  ```

- **Run Linters** (Flake8 & Pylint):
  ```bash
  make lint
  ```

- **Run Security Checks** (Bandit):
  ```bash
  make security
  ```

## Usage Example (API)

Once the backend API is running, you can predict a customer's churn risk using the `/predict` endpoint:

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{
          "CreditScore": 600,
          "Age": 40,
          "Tenure": 3,
          "Balance": 60000.0,
          "NumOfProducts": 2,
          "HasCrCard": 1,
          "IsActiveMember": 1,
          "EstimatedSalary": 50000.0,
          "Gender": 0
         }'
```
