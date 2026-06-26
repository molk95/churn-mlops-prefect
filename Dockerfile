FROM python:3.10-slim

ENV GIT_PYTHON_REFRESH=quiet

WORKDIR /app

# Install system dependencies if any are needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . .

# Ensure writable temp dirs for matplotlib config and mlruns
RUN mkdir -p /tmp/matplotlib /app/mlruns && chmod -R 777 /tmp/matplotlib /app/mlruns

# Expose port 8080 for FastAPI
EXPOSE 8080

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
