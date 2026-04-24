"""
Evaluate model metric and promote/fallback in MLFlow Model Registry

Reads:  RUN_ID, MODEL_NAME from env
Writes: deploy_stage, deploy_version  → $GITHUB_OUTPUT  (job outputs)
                                      → $GITHUB_ENV      (same-job env vars)
"""

import os
import sys
import time
from mlflow.tracking import MlflowClient

# ------
# Config
# ------
run_id = os.environ.get("RUN_ID")
model_name = os.environ.get("MODEL_NAME", "credit_risk_model")
threshold = 0.78
github_output = os.environ.get("GITHUB_OUTPUT")
github_env = os.environ.get("GITHUB_ENV")

client = MlflowClient()


def set_outputs(stage: str, version: str) -> None:
    """Write deploy outputs to GITHUB_OUTPUT (step output) and GITHUB_ENV"""
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"deploy_stage={stage}\n")
            f.write(f"deploy_version={version}\n")
    if github_env:
        with open(github_env, "a") as f:
            f.write(f"DEPLOY_STAGE={stage}\n")
            f.write(f"DEPLOY_VERSION={version}\n")


# ----------
# GET METRIC
# ----------
print("Evaluating model metrics...")
run = client.get_run(run_id)
roc_auc = run.data.metrics.get("roc_auc", 0)

print(f"Run ID:  {run_id}")
print(f"ROC-AUC: {roc_auc}")

# -------------
# WAIT REGISTRY
# -------------
print("Waiting for model version to be registered...")
target_version = None

for i in range(10):
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.run_id == run_id:
            target_version = v
            break
    if target_version:
        print(f"Model version found: {target_version.version}")
        break
    print(f"Retry {i + 1}/10 - model not found yet...")
    time.sleep(3)

if target_version is None:
    print("ERROR: Model version NOT found after retry")
    sys.exit(1)

model_version = target_version.version

# --------------
# DECISION LOGIC
# --------------
if roc_auc > threshold:
    print(f"Model PASSED threshold ({roc_auc} > {threshold})")

    # Archive old Production
    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production":
            client.transition_model_version_stage(
                name=mv.name,
                version=mv.version,
                stage="Archived",
            )
            print(f"Archived old Production v{mv.version}")

    # Promote new model
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production",
    )
    print(f"Promoted v{model_version} to Production")
    set_outputs("Production", model_version)

else:
    print(f"Model FAILED threshold ({roc_auc} <= {threshold})")

    # Fallback ke existing Production
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            prod_version = prod_versions[0].version
            print(f"Using existing Production v{prod_version}")
            set_outputs("Production", prod_version)
        else:
            print("ERROR: No Production model exists")
            sys.exit(1)
    except Exception as e:
        print("ERROR accessing Production model:", str(e))
        sys.exit(1)

print("PROMOTION STEP DONE...")
