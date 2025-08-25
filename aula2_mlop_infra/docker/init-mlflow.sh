#!/bin/bash
# init-mlflow.sh

# This script ensures the MLflow directory has the correct permissions on startup.

# Create the directories inside the mounted volume, if they don't exist.
mkdir -p /mlflow/db
mkdir -p /artifacts

# Set full read/write/execute permissions for the entire /mlflow and /artifacts directories.
# This is the key step to prevent "Permission Denied" errors.
chmod -R 777 /mlflow
chmod -R 777 /artifacts

# Now, execute the main command: start the MLflow server.
# The 'exec' command replaces the script process with the mlflow server process.
exec mlflow server \
    --backend-store-uri sqlite:////mlflow/db/mlflow.db \
    --artifacts-destination /artifacts \
    --serve-artifacts \
    --host 0.0.0.0 \
    --port 5000
