"""
MLOps Pipeline DAG - Integrates with GitHub Actions
This DAG demonstrates how Airflow can orchestrate MLOps workflows
triggered by GitHub Actions CI/CD pipeline.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'mlops_github_integration_pipeline',
    default_args=default_args,
    description='MLOps pipeline integrated with GitHub Actions',
    schedule='@daily',  # Can be triggered by GitHub Actions
    catchup=False,
    max_active_runs=1,
    tags=['mlops', 'github-actions', 'ml-training', 'model-registry'],
)

def check_mlflow_connection(**context):
    """Check if MLflow tracking server is accessible"""
    try:
        # Set MLflow tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Create MLflow client and test connection
        client = MlflowClient()
        experiments = client.search_experiments()
        
        logger.info(f"✅ MLflow connection successful. Found {len(experiments)} experiments.")
        return True
    except Exception as e:
        logger.error(f"❌ MLflow connection failed: {e}")
        raise

def prepare_training_data(**context):
    """Prepare synthetic bonsai dataset for training"""
    logger.info("📊 Preparing training data...")
    
    # Create synthetic bonsai dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_classes=4,
        n_informative=4,
        n_redundant=0,
        random_state=42
    )
    
    # Feature names for bonsai classification
    feature_names = ['leaf_length_cm', 'leaf_width_cm', 'branch_thickness_mm', 'height_cm']
    species = ['Juniper', 'Ficus', 'Pine', 'Maple']
    
    # Scale features to realistic measurements
    X[:, 0] = X[:, 0] * 0.5 + 2.0  # leaf_length: 1.5-2.5 cm
    X[:, 1] = X[:, 1] * 0.3 + 1.5  # leaf_width: 1.2-1.8 cm
    X[:, 2] = X[:, 2] * 2.0 + 5.0  # branch_thickness: 3-7 mm
    X[:, 3] = X[:, 3] * 10.0 + 25.0  # height: 15-35 cm
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Store in XCom for next tasks
    context['task_instance'].xcom_push(key='X_train', value=X_train.tolist())
    context['task_instance'].xcom_push(key='X_test', value=X_test.tolist())
    context['task_instance'].xcom_push(key='y_train', value=y_train.tolist())
    context['task_instance'].xcom_push(key='y_test', value=y_test.tolist())
    context['task_instance'].xcom_push(key='feature_names', value=feature_names)
    context['task_instance'].xcom_push(key='species', value=species)
    
    logger.info(f"✅ Data prepared: {X_train.shape[0]} training, {X_test.shape[0]} test samples")
    return {"train_samples": X_train.shape[0], "test_samples": X_test.shape[0]}

def train_model_experiment(**context):
    """Train multiple model configurations and log to MLflow"""
    logger.info("🤖 Starting model training experiments...")
    
    # Get data from XCom
    X_train = np.array(context['task_instance'].xcom_pull(key='X_train'))
    X_test = np.array(context['task_instance'].xcom_pull(key='X_test'))
    y_train = np.array(context['task_instance'].xcom_pull(key='y_train'))
    y_test = np.array(context['task_instance'].xcom_pull(key='y_test'))
    
    # Set MLflow tracking URI
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Set MLflow experiment
    experiment_name = "Airflow-Bonsai-Classification"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    # Model configurations to test
    configs = [
        {"name": "baseline", "n_estimators": 50, "max_depth": 3},
        {"name": "balanced", "n_estimators": 100, "max_depth": 5},
        {"name": "complex", "n_estimators": 200, "max_depth": 8},
    ]
    
    best_model_info = {"accuracy": 0, "run_id": None, "model_name": None}
    
    for config in configs:
        with mlflow.start_run(run_name=f"airflow_{config['name']}_model"):
            # Train model
            model = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log parameters and metrics
            mlflow.log_params({
                "n_estimators": config['n_estimators'],
                "max_depth": config['max_depth'],
                "model_type": "RandomForestClassifier",
                "triggered_by": "airflow_github_actions"
            })
            
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            })
            
            # Log model with signature
            signature = mlflow.models.infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                input_example=X_train[:5]
            )
            
            # Track best model
            if accuracy > best_model_info["accuracy"]:
                best_model_info.update({
                    "accuracy": accuracy,
                    "run_id": mlflow.active_run().info.run_id,
                    "model_name": config['name']
                })
            
            logger.info(f"✅ {config['name']} model: Accuracy={accuracy:.3f}")
    
    # Store best model info in XCom
    context['task_instance'].xcom_push(key='best_model_info', value=best_model_info)
    
    return best_model_info

def register_best_model(**context):
    """Register the best performing model in MLflow Model Registry"""
    logger.info("📦 Registering best model...")
    
    best_model_info = context['task_instance'].xcom_pull(key='best_model_info')
    
    if not best_model_info or not best_model_info.get('run_id'):
        raise ValueError("No best model information found")
    
    # Set MLflow tracking URI
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Register model
    model_name = "Airflow-Bonsai-Classifier-Production"
    model_uri = f"runs:/{best_model_info['run_id']}/model"
    
    try:
        # Check if model exists in registry
        client = MlflowClient()
        try:
            registered_model = client.get_registered_model(model_name)
            logger.info(f"Model {model_name} already exists in registry")
        except mlflow.exceptions.RestException:
            # Model doesn't exist, create it
            logger.info(f"Creating new registered model: {model_name}")
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            description=f"Best bonsai classifier from Airflow pipeline. "
                       f"Accuracy: {best_model_info['accuracy']:.3f}"
        )
        
        logger.info(f"✅ Model registered: {model_name} v{model_version.version}")
        
        # Store registration info
        registration_info = {
            "model_name": model_name,
            "version": model_version.version,
            "accuracy": best_model_info['accuracy'],
            "run_id": best_model_info['run_id']
        }
        
        context['task_instance'].xcom_push(key='registration_info', value=registration_info)
        return registration_info
        
    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        raise

def validate_deployment(**context):
    """Validate that the model is properly deployed and accessible"""
    logger.info("🔍 Validating model deployment...")
    
    registration_info = context['task_instance'].xcom_pull(key='registration_info')
    
    # Set MLflow tracking URI
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    
    # Try to load the registered model
    try:
        model_name = registration_info['model_name']
        model_version = registration_info['version']
        
        # Load model from registry using the new format
        model_uri = f"models:/{model_name}/{model_version}"
        loaded_model = mlflow.sklearn.load_model(model_uri)
        
        # Test prediction with sample data
        test_features = [[2.0, 1.5, 5.0, 25.0]]  # Sample bonsai measurements
        prediction = loaded_model.predict(test_features)
        prediction_proba = loaded_model.predict_proba(test_features)
        
        logger.info(f"✅ Model validation successful. Sample prediction: {prediction[0]}")
        logger.info(f"Prediction probabilities: {prediction_proba[0]}")
        
        validation_result = {
            "status": "success",
            "model_name": model_name,
            "model_version": model_version,
            "sample_prediction": int(prediction[0]),
            "prediction_confidence": float(max(prediction_proba[0]))
        }
        
        return validation_result
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {e}")
        raise

def send_pipeline_notification(**context):
    """Send notification about pipeline completion"""
    registration_info = context['task_instance'].xcom_pull(key='registration_info')
    
    message = f"""
    🎉 MLOps Pipeline Completed Successfully!
    
    📊 Model: {registration_info['model_name']}
    🔢 Version: {registration_info['version']}
    🎯 Accuracy: {registration_info['accuracy']:.3f}
    🚀 Triggered by: GitHub Actions + Airflow
    
    🌐 MLflow UI: http://localhost:5000
    📋 Airflow UI: http://localhost:8080
    """
    
    logger.info(message)
    print(message)  # This will appear in Airflow logs
    
    return {"status": "notification_sent", "message": message}

# Task 1: Check MLflow connection
check_mlflow_task = PythonOperator(
    task_id='check_mlflow_connection',
    python_callable=check_mlflow_connection,
    dag=dag,
)

# Task 2: Wait for MLflow to be healthy (using simple HTTP check)
wait_for_mlflow = PythonOperator(
    task_id='wait_for_mlflow_health',
    python_callable=check_mlflow_connection,
    dag=dag,
)

# Task 3: Prepare training data
prepare_data_task = PythonOperator(
    task_id='prepare_training_data',
    python_callable=prepare_training_data,
    dag=dag,
)

# Task 4: Train models
train_model_task = PythonOperator(
    task_id='train_model_experiments',
    python_callable=train_model_experiment,
    dag=dag,
)

# Task 5: Register best model
register_model_task = PythonOperator(
    task_id='register_best_model',
    python_callable=register_best_model,
    dag=dag,
)

# Task 6: Validate deployment
validate_deployment_task = PythonOperator(
    task_id='validate_deployment',
    python_callable=validate_deployment,
    dag=dag,
)

# Task 7: Send notification
notification_task = PythonOperator(
    task_id='send_pipeline_notification',
    python_callable=send_pipeline_notification,
    dag=dag,
)

# Task 8: Cleanup (optional)
cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command="""
    echo "🧹 Cleaning up temporary files..."
    # Add any cleanup commands here
    echo "✅ Cleanup completed"
    """,
    dag=dag,
)

# Define task dependencies
check_mlflow_task >> wait_for_mlflow >> prepare_data_task
prepare_data_task >> train_model_task >> register_model_task
register_model_task >> validate_deployment_task >> notification_task
notification_task >> cleanup_task
