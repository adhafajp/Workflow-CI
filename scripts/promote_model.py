"""
Evaluate model metric and promote/fallback in MLFlow Model Registry

Reads:  RUN_ID, MODEL_NAME, COMMIT_MESSAGE from env
Writes: deploy_stage, deploy_version, deploy_semver -> $GITHUB_OUTPUT  (job outputs)
                                                    -> $GITHUB_ENV      (same-job env vars)
"""

import os
import sys
import time
import re
from mlflow.tracking import MlflowClient

# ------
# Config
# ------
run_id = os.environ.get("RUN_ID")
model_name = os.environ.get("MODEL_NAME", "credit_risk_model")
commit_message = os.environ.get("COMMIT_MESSAGE", "").lower()
threshold = 0.78
github_output = os.environ.get("GITHUB_OUTPUT")
github_env = os.environ.get("GITHUB_ENV")

client = MlflowClient()


def set_outputs(stage: str, version: str, semver: str) -> None:
    """Write deploy outputs to GITHUB_OUTPUT (step output) and GITHUB_ENV"""
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"deploy_stage={stage}\n")
            f.write(f"deploy_version={version}\n")
            f.write(f"deploy_semver={semver}\n")
    if github_env:
        with open(github_env, "a") as f:
            f.write(f"DEPLOY_STAGE={stage}\n")
            f.write(f"DEPLOY_VERSION={version}\n")
            f.write(f"DEPLOY_SEMVER={semver}\n")


def get_bump_type(msg: str) -> str:
    """
    Determine semantic version bump type based on Conventional Commits
    """
    if "breaking change" in msg or re.match(r"^\w+!:", msg):
        return "major"
    elif msg.startswith("feat:"):
        return "minor"
    elif msg.startswith("fix:") or msg.startswith("perf:"):
        return "patch"
    elif msg.startswith("chore:") or msg.startswith("ci:"):
        return "none"
    else:
        return "patch"


def calculate_new_semver(current_semver: str, bump_type: str) -> str:
    """
    Calculate new semantic version string: MAJOR.MINOR.PATCH
    """
    try:
        major, minor, patch = map(int, current_semver.split("."))
    except ValueError:
        major, minor, patch = 1, 0, 0

    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    elif bump_type == "none":
        return current_semver


def get_latest_production_semver(model_name: str) -> str:
    """
    Retrieve semver tag from the current Production model in registry
    Fallback to '1.0.0' if no production model exists
    """
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        if prod_versions:
            semver_tag = prod_versions[0].tags.get("semver")
            if semver_tag:
                return semver_tag
    except Exception:
        pass
    
    return "1.0.0"


# ----------
# GET METRIC
# ----------
print("Evaluating model metrics...")
run = client.get_run(run_id)
roc_auc = run.data.metrics.get("roc_auc", 0)

print(f"Run ID:     {run_id}")
print(f"ROC-AUC:    {roc_auc}")
print(f"Commit Msg: {commit_message}")

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

    # Determine New SemVer
    current_semver = get_latest_production_semver(model_name)
    bump_type = get_bump_type(commit_message)
    new_semver = calculate_new_semver(current_semver, bump_type)
    
    print(f"Bump Type: {bump_type.upper()} | Old: {current_semver} -> New: {new_semver}")

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

    # Set semver tag
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="semver",
        value=new_semver,
    )

    print(f"Promoted v{model_version} to Production (semver: {new_semver})")
    set_outputs("Production", model_version, new_semver)

else:
    print(f"Model FAILED threshold ({roc_auc} <= {threshold})")

    # Fallback ke existing Production
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])

        if prod_versions:
            prod_mv = prod_versions[0]
            existing_semver = prod_mv.tags.get("semver", "1.0.0")

            print(f"Using existing Production v{prod_mv.version} (semver: {existing_semver})")
            set_outputs("Production", prod_mv.version, existing_semver)

        else:
            print("ERROR: No Production model exists")
            sys.exit(1)

    except Exception as e:
        print("ERROR accessing Production model:", str(e))
        sys.exit(1)

print("PROMOTION STEP DONE...")