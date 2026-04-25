# ================
# Stage 1: Builder
# ================
FROM python:3.12.7 AS builder

WORKDIR /build

COPY requirements-serve.txt .

# Install ke prefix terpisah biar bisa di copy ke stage 2
RUN pip install --no-cache-dir --prefix=/install -r requirements-serve.txt

# =====================
# Stage 2: Final (slim)
# =====================
FROM python:3.12.7-slim

WORKDIR /app

# Copy hasil install dari stage 1 — tanpa pip cache, build tool, dll
COPY --from=builder /install /usr/local

# Copy preprocessing utils
COPY preprocessing_utils.py .

# Copy preprocessor artifact
COPY credit_risk_preprocessing/preprocessor.pkl ./credit_risk_preprocessing/

# Copy model yang sudah di download di CI
COPY model_artifact/ ./model/

EXPOSE 8080

CMD ["mlflow", "models", "serve", \
     "-m", "/app/model", \
     "-p", "8080", \
     "--host", "0.0.0.0", \
     "--env-manager=local"]