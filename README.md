# Workflow CI - Credit Risk Model

## Overview
Repository untuk submission IDCamp 2025 Dicoding MLOps Expert - Kriteria 3

## Structure
Workflow-CI/
├── .github/workflows/ci.yml # CI/CD Pipeline
└── MLProject/
├── MLproject # MLflow project definition
├── conda.yaml # Environment
├── modelling.py # Training script
└── credit_risk_preprocessing/ # Dataset

## Feature
- Automated training on push
- Hyperparameter tuning with GridSearchCV
- Remote tracking to DagsHub
- Artifact upload to Google Drive
- Docker image build & push to Docker Hub
- Model versioning with run ID

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
docker pull adhafajp/credit-risk-model:latest
docker run -p 5000:8080 adhafajp/credit-risk-model:latest

---

## **GitHub Secrets Configuration**

Buka repository GitHub → Settings → Secrets and variables → Actions → New repository secret:

| Secret Name | Value | Description |
|-------------|-------|-------------|
| `DAGSHUB_USERNAME` | `adhafajp` | DagsHub username |
| `DAGSHUB_TOKEN` | `dp1_xxxx...` | DagsHub access token |
| `DOCKERHUB_USERNAME` | `your_dockerhub_username` | Docker Hub username |
| `DOCKERHUB_TOKEN` | `xxxx-xxxx-xxxx` | Docker Hub access token |
| `GOOGLE_OAUTH_CREDENTIALS` | `{...JSON...}` | Google OAuth JSON |
| `GOOGLE_DRIVE_FOLDER_ID` | `1abcXYZ...` | Google Drive folder ID |

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
mkdir -p .github/workflows MLProject/credit_risk_preprocessing

# Copy dataset
cp -r ../Membangun_model/credit_risk_preprocessing/* MLProject/credit_risk_preprocessing/

# Add all file
git add .
git commit -m "Initial commit: MLflow CI/CD pipeline"
git push origin main
```

## 3. Setup Secrets
Go to GitHub → Settings → Secrets → Add all 6 secrets

## 4. Test Pipeline
```bash
# Push a change to trigger
git commit --allow-empty -m "Trigger CI pipeline"
git push
```

## 5. Verify Results
- GitHub Actions: Work

- DagsHub: New experiment appears

- Google Drive: Artifact uploaded

- Docker Hub: New image pushed

## Author

**Adhafa J.P.**  
AI Engineer Student | MLOps Enthusiast  

- GitHub: https://github.com/adhafajp  
- DagsHub: https://dagshub.com/adhafajp  
- Docker Hub: https://hub.docker.com/u/adhafajp  

---

### About This Project
This project is developed as part of the IDCamp 2025 Dicoding MLOps Expert submission, focusing on end-to-end machine learning pipeline automation using MLflow, GitHub Actions, and Docker.