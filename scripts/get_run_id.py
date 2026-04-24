"""
Get the latest MLflow run_id from the credit_risk_ci experiment
Prints the run_id (digunakan untuk capture in CI step)
"""

import mlflow

runs = mlflow.search_runs(
    experiment_names=["credit_risk_ci"],
    max_results=1,
    order_by=["start_time DESC"],
)

print(runs["run_id"].iloc[0])
