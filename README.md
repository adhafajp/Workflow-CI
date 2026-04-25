# Workflow CI - Credit Risk Model

## Overview
Repository untuk submission IDCamp 2025 Dicoding MLOps Expert - Kriteria 3

## Structure
```text
Workflow-CI/
├── .github/
│   ├── workflows/
│   │   ├── pipeline.yml           # Orchestrator (entry point)
│   │   ├── train.yml              # Module: Training
│   │   ├── validate.yml           # Module: Inference validation
│   │   ├── promote.yml            # Module: MLflow registry & promotion
│   │   ├── artifact.yml           # Module: Google Drive upload
│   │   └── docker.yml             # Module: Docker build & push
│   └── actions/
│       ├── setup-python/          # Composite: setup Python + env check
│       ├── install-deps/          # Composite: install requirements.txt
│       └── mlflow-setup/          # Composite: set DagsHub tracking URI
├── scripts/
│   ├── get_run_id.py              # Ambil latest run_id dari MLflow
│   └── promote_model.py           # Evaluate metric & promote ke Production
└── MLProject/
    ├── MLproject                  # MLflow project definition
    ├── conda.yaml                 # Environment
    ├── modelling.py               # Training script
    └── credit_risk_preprocessing/ # Dataset
```

## Pipeline DAG
```text
train → validate → promote → artifact → docker
```
Setiap stage adalah reusable workflow yang bisa di debug dan di rerun secara independen.

## Feature
- Automated training on push
- Hyperparameter tuning with GridSearchCV
- Remote tracking to DagsHub
- Inference validation sebelum promote
- Model versioning & promotion via MLflow Registry
- Artifact upload to Google Drive
- Docker image build & push to Docker Hub
- Modular pipeline (reusable workflows + composite actions)

## Docker Hub
https://hub.docker.com/repository/docker/adhafajp/credit-risk-model/general

## DagsHub
https://dagshub.com/adhafajp/credit_risk_model/experiments

## Run Locally
```bash
cd MLProject
# Development
mlflow run . --env-manager=local
# ATAU
# Reproducible
mlflow run .
```

## Run with Docker
```bash
docker pull adhafajp/credit-risk-model:latest
docker run -p 5000:8080 adhafajp/credit-risk-model:latest
```

---

## **GitHub Secrets & Variables Configuration**

Buka repository GitHub → Settings → Secrets and variables → Actions

### Secrets
Tab **Secrets** → New repository secret — untuk data sensitif:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `DAGSHUB_TOKEN` | `dp1_xxxx...` | DagsHub access token |
| `DOCKERHUB_TOKEN` | `xxxx-xxxx-xxxx` | Docker Hub access token |
| `GOOGLE_OAUTH_CREDENTIALS` | `{...JSON...}` | Google OAuth JSON |
| `GOOGLE_DRIVE_FOLDER_ID` | `1abcXYZ...` | Google Drive folder ID |

### Variables
Tab **Variables** → New repository variable:

| Variable Name | Value | Description |
|---------------|-------|-------------|
| `DAGSHUB_USERNAME` | `adhafajp` | DagsHub username |
| `DOCKERHUB_USERNAME` | `adhafajp` | Docker Hub username |

---

## **GitHub Actions Settings**

Buka repository GitHub → Settings → Actions → General → Workflow permissions:

Pastikan diset ke **"Read and write permissions"** agar `actions/upload-artifact` dan `actions/download-artifact` bisa beroperasi antar job.

---

## **Step-by-Step Execution**

### **1. Create GitHub Repository**
```bash
# Clone locally
git clone https://github.com/adhafajp/Workflow-CI.git
cd Workflow-CI
```

### 2. Add Files
```bash
# Create folder structure
mkdir -p .github/workflows .github/actions/setup-python
mkdir -p .github/actions/install-deps .github/actions/mlflow-setup
mkdir -p scripts MLProject/credit_risk_preprocessing

# Copy dataset
cp -r ../Membangun_model/credit_risk_preprocessing/* MLProject/credit_risk_preprocessing/

# Add all file
git add .
git commit -m "Initial commit: modular MLflow CI/CD pipeline"
git push origin main
```

### 3. Setup Secrets
Go to GitHub → Settings → Secrets → Add all 6 secrets

### 4. Check Actions Permissions
Go to GitHub → Settings → Actions → General → set **"Read and write permissions"**

### 5. Test Pipeline
```bash
# Push a change to trigger
git commit --allow-empty -m "Trigger CI pipeline"
git push
```

### 6. Verify Results
- GitHub Actions: Semua 5 jobs (train → validate → promote → artifact → docker) passed

- DagsHub: New experiment appears

- Google Drive: Artifact uploaded

- Docker Hub: New image pushed

## Author

**Adhafa J.P.**

*AI Engineer Student | MLOps & ML Enthusiast*

- [GitHub](https://github.com/adhafajp)
- [DagsHub](https://dagshub.com/adhafajp)
- [Docker Hub](https://hub.docker.com/u/adhafajp)

---

### About This Project
This project is developed as part of the IDCamp 2025 Dicoding MLOps Expert submission, focusing on end-to-end machine learning pipeline automation using MLflow, GitHub Actions, and Docker