#!/bin/bash
# ==================================================================
# Inner Loop Script
# Usage: bash dev.sh [preprocess] [train] [build] [run] [test] [all]
# ==================================================================

set -e

IMAGE_NAME="credit-risk-model:local"
PORT=5000

# ==========
# PREPROCESS
# ==========
preprocess() {
  echo ">>> [1/4] Running preprocessing..."
  cd "$(dirname "$0")"
  python automate_preprocess.py
  echo ">>> Preprocessing done"
}

# =====
# TRAIN
# =====
train() {
  echo ">>> [2/4] Running training..."
  cd "$(dirname "$0")"
  mlflow run . \
    --experiment-name credit_risk_local \
    --run-name "local_run_$(date +'%Y%m%d_%H%M%S')" \
    --env-manager=local
  echo ">>> Training done"
}

# ===================
# COPY MODEL ARTIFACT
# ===================
copy_artifact() {
  echo ">>> Preparing model_artifact/ for Docker build..."
  SCRIPT_DIR="$(dirname "$0")"
  
  # Search new run dari MLflow local
  MLRUNS_DIR="$SCRIPT_DIR/mlruns"
  if [ ! -d "$MLRUNS_DIR" ]; then
    echo "ERROR: mlruns/ not found. Run training first."
    exit 1
  fi

  # Search model artifact dari new run
  LATEST_MODEL=$(find "$MLRUNS_DIR" -type d -name "model" | sort -t'/' -k8 | tail -1)
  
  if [ -z "$LATEST_MODEL" ]; then
    echo "ERROR: No model artifact found in mlruns/"
    exit 1
  fi

  echo "Found model at: $LATEST_MODEL"
  rm -rf "$SCRIPT_DIR/model_artifact"
  cp -r "$(dirname "$LATEST_MODEL")" "$SCRIPT_DIR/model_artifact"
  echo ">>> model_artifact/ ready"
}

# ============
# BUILD DOCKER
# ============
build() {
  echo ">>> [3/4] Building Docker image: $IMAGE_NAME"
  SCRIPT_DIR="$(dirname "$0")"
  copy_artifact
  docker build \
    --no-cache \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$SCRIPT_DIR/"
  echo ">>> Docker build done"
}

# ===========
# RUN DOCKER
# ===========
run_container() {
  echo ">>> [4/4] Running container on port $PORT..."
  docker stop credit-risk-local 2>/dev/null || true
  docker rm credit-risk-local 2>/dev/null || true
  docker run -d \
    --name credit-risk-local \
    -p $PORT:8080 \
    "$IMAGE_NAME"
  echo ">>> Container running at http://localhost:$PORT"
  echo ">>> Logs: docker logs -f credit-risk-local"
}

# ====
# TEST
# ====
test_inference() {
  echo ">>> Testing inference..."

  echo "Waiting for server..."
  for i in {1..20}; do
    if curl -s http://localhost:$PORT/ping > /dev/null 2>&1; then
      echo "Server ready"
      break
    fi
    sleep 2
  done

  echo ""
  echo "--- Expect Predict 0 ---"
  curl -s -X POST http://localhost:$PORT/invocations \
    -H "Content-Type: application/json" \
    -d '{
      "dataframe_split": {
        "columns": ["person_age","person_income","person_emp_length","loan_grade",
                    "loan_amnt","loan_int_rate","loan_percent_income",
                    "cb_person_cred_hist_length","person_home_ownership",
                    "loan_intent","cb_person_default_on_file"],
        "data": [[45, 90000, 10, "A", 5000, 7.5, 0.06, 12, "MORTGAGE", "EDUCATION", "N"]]
      }
    }' | python3 -m json.tool

  echo ""
  echo "--- Expect Predict 1 ---"
  curl -s -X POST http://localhost:$PORT/invocations \
    -H "Content-Type: application/json" \
    -d '{
      "dataframe_split": {
        "columns": ["person_age","person_income","person_emp_length","loan_grade",
                    "loan_amnt","loan_int_rate","loan_percent_income",
                    "cb_person_cred_hist_length","person_home_ownership",
                    "loan_intent","cb_person_default_on_file"],
        "data": [[21, 8000, 0, "G", 14000, 23.0, 0.59, 1, "RENT", "PERSONAL", "Y"]]
      }
    }' | python3 -m json.tool
}

# ====
# MAIN
# ====
case "${1:-all}" in
  preprocess) preprocess ;;
  train)      train ;;
  build)      build ;;
  run)        run_container ;;
  test)       test_inference ;;
  all)
    preprocess
    train
    build
    run_container
    test_inference
    ;;
  *)
    echo "Usage: bash dev.sh [preprocess|train|build|run|test|all]"
    exit 1
    ;;
esac