# Churn Analytics MLOps Pipeline

This project is an end-to-end MLOps pipeline for predicting customer churn. It includes data preparation, model training orchestration, a REST API for inference, an interactive Streamlit dashboard for prediction monitoring, and automated multi-container deployment using Docker and GitHub Actions.

---

## Features

- **Data Pipeline**: Automated data preparation, cleaning, and scaling using `StandardScaler`.
- **Model Training**: Support for training multiple models (Random Forest, Logistic Regression, Gradient Boosting) via dynamic endpoints.
- **REST API**: Built with FastAPI to handle preparation, training, and inference.
- **Interactive UI**: A Streamlit dashboard to test individual customer churn, monitor training, and download models.
- **Orchestration**: Prefect workflow orchestration for local and remote pipelines.
- **Experiment Tracking**: MLflow tracking utilizing SQLite database (`sqlite:///mlflow.db`) with parameter alignment for parallel coordinates plots.
- **Multi-Container Docker Stack**: Orchestrated deployment of backend API, frontend UI, and MLflow server using Docker Compose.
- **GitHub Actions CI/CD**: Automatic linting, code quality auditing, unit testing, and Docker Hub image pushing on repository updates.

---

## Project Structure

* `app.py`: FastAPI serving backend application.
* `app_streamlit.py`: Streamlit frontend application.
* `model_pipeline.py`: Core machine learning logic (preprocessing, training, evaluation, saving).
* `pipeline_prefect.py` / `deploiement_prefect.py`: Prefect workflow orchestration tasks and schedules.
* `Dockerfile` / `Dockerfile.ui` / `Dockerfile.mlflow`: Container configurations for API, UI, and MLflow.
* `docker-compose.yml`: Multi-container local orchestration stack.
* `.github/workflows/ci-cd.yml`: GitHub Actions automated workflow pipeline.
* `Makefile`: Automates setup, testing, formatting, linting, and Docker operations.

---

## Installation & Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/molk95/churn-mlops-prefect.git
   cd churn-mlops-prefect
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   make install
   ```

---

## Running the Services

### Local (Bare Metal)

You can run each component using the provided `Makefile`:

1. **Start MLflow Server**:
   ```bash
   make run-mlflow
   ```
   *Available at `http://localhost:5000`.*

2. **Start Backend API (FastAPI)**:
   ```bash
   make run-api
   ```
   *Available at `http://localhost:8080` (Swagger docs at `/docs`).*

3. **Start Frontend Dashboard (Streamlit)**:
   ```bash
   make run-ui
   ```
   *Available at `http://localhost:8501`.*

4. **Start Prefect Orchestration Server**:
   ```bash
   make run-prefect
   ```
   *Runs locally on `http://localhost:4200`.*

5. **Run the Prefect Pipeline Flow**:
   ```bash
   make prefect-all
   ```

---

### Local (Docker & Docker Compose)

To spin up the entire MLOps stack (API, Streamlit UI, MLflow) in Docker containers with persistent local volume mapping:

* **Start the Multi-Container Stack**:
  ```bash
  make compose-up
  ```

* **Stop the Stack**:
  ```bash
  make compose-down
  ```

---

## Testing & Quality Assurance

* **Run Unit Tests**:
  ```bash
  make test
  ```
* **Auto-format Code (Black)**:
  ```bash
  make format
  ```
* **Code Linting (Flake8 & Pylint)**:
  ```bash
  make lint
  ```
* **Security Auditing (Bandit)**:
  ```bash
  make security
  ```

---

## Docker Hub Publishing

Build, tag, and push local images to Docker Hub under the username `molksaouabi`:

* **Build Images**:
  ```bash
  make docker-build
  ```
* **Tag and Push to Docker Hub**:
  ```bash
  make docker-push
  ```
* **Run Containerized Services Individually**:
  ```bash
  make docker-run
  ```

---

## CI/CD with GitHub Actions

The repository includes a GitHub Actions workflow in `.github/workflows/ci-cd.yml` which executes on every push to the `main` branch. 

### Pipeline Stages:
1. **Lint & Format Check**: Executes code syntax rules (`black`, `flake8`).
2. **Security Scan**: Checks for vulnerabilities using `bandit`.
3. **Unit Tests**: Asserts correct application behavior via `pytest`.
4. **Build & Push**: Compiles the Docker images and pushes them to Docker Hub (`molksaouabi/molk_saouabi_sde_2_mlops_api`, `molksaouabi/molk_saouabi_sde_2_mlops_ui`, and `molksaouabi/molk_saouabi_sde_2_mlops_mlflow`).

*Note: Ensure you add `DOCKER_HUB_USERNAME` and `DOCKER_HUB_ACCESS_TOKEN` as Repository Secrets in your GitHub repository settings.*
