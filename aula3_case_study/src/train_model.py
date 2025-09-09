"""
Automated Model Training Script
Triggered by GitHub Actions, orchestrated by Airflow
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import numpy as np
import logging
import os
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_bonsai_dataset():
    """Create synthetic bonsai species classification dataset"""
    logger.info("📊 Creating bonsai dataset...")
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1500,
        n_features=4,
        n_classes=4,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Feature names for bonsai classification
    feature_names = ['leaf_length_cm', 'leaf_width_cm', 'branch_thickness_mm', 'height_cm']
    species_names = ['Juniper', 'Ficus', 'Pine', 'Maple']
    
    # Scale features to realistic bonsai measurements
    X[:, 0] = X[:, 0] * 0.5 + 2.0    # leaf_length: 1.5-2.5 cm
    X[:, 1] = X[:, 1] * 0.3 + 1.5    # leaf_width: 1.2-1.8 cm
    X[:, 2] = X[:, 2] * 2.0 + 5.0    # branch_thickness: 3-7 mm
    X[:, 3] = X[:, 3] * 10.0 + 25.0  # height: 15-35 cm
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [species_names[i] for i in y]
    df['species_id'] = y
    
    logger.info(f"✅ Dataset created: {len(df)} samples, {len(feature_names)} features")
    return df, feature_names, species_names

def train_model_variants(X_train, X_test, y_train, y_test, experiment_name="Automated-Bonsai-Training"):
    """Train multiple model variants and compare performance"""
    
    # Set up MLflow experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            mlflow.create_experiment(experiment_name)
    except:
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    # Model configurations to test
    model_configs = [
        {
            "name": "lightweight_model",
            "n_estimators": 50,
            "max_depth": 3,
            "min_samples_split": 5,
            "description": "Fast inference model for edge deployment"
        },
        {
            "name": "balanced_model",
            "n_estimators": 100,
            "max_depth": 5,
            "min_samples_split": 3,
            "description": "Balanced accuracy and speed"
        },
        {
            "name": "high_accuracy_model",
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 2,
            "description": "Maximum accuracy model"
        },
        {
            "name": "robust_model",
            "n_estimators": 150,
            "max_depth": 6,
            "min_samples_split": 4,
            "description": "Robust model with good generalization"
        }
    ]
    
    best_model_info = None
    best_accuracy = 0.0
    
    logger.info(f"🤖 Training {len(model_configs)} model variants...")
    
    for config in model_configs:
        with mlflow.start_run(run_name=f"github_actions_{config['name']}"):
            logger.info(f"Training {config['name']}...")
            
            # Train model
            model = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                max_depth=config['max_depth'],
                min_samples_split=config['min_samples_split'],
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate detailed metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log parameters
            mlflow.log_params({
                "model_name": config['name'],
                "n_estimators": config['n_estimators'],
                "max_depth": config['max_depth'],
                "min_samples_split": config['min_samples_split'],
                "model_type": "RandomForestClassifier",
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "triggered_by": "github_actions",
                "timestamp": datetime.now().isoformat(),
                "description": config['description']
            })
            
            # Log metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "training_score": model.score(X_train, y_train)
            })
            
            # Log model with signature
            signature = mlflow.models.infer_signature(X_train, y_train)
            mlflow.sklearn.log_model(
                model, 
                "model",
                signature=signature,
                input_example=X_test[:5]
            )
            
            # Log classification report as artifact
            class_report = classification_report(y_test, y_pred, output_dict=True)
            mlflow.log_dict(class_report, "classification_report.json")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_info = {
                    "run_id": mlflow.active_run().info.run_id,
                    "model_name": config['name'],
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                    "config": config
                }
            
            logger.info(f"✅ {config['name']}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    
    return best_model_info

def register_best_model(best_model_info, model_registry_name="GitHub-Actions-Bonsai-Classifier"):
    """Register the best performing model in MLflow Model Registry"""
    
    if not best_model_info:
        raise ValueError("No best model information provided")
    
    logger.info(f"📦 Registering best model: {best_model_info['model_name']}")
    
    # Model URI from the best run
    model_uri = f"runs:/{best_model_info['run_id']}/model"
    
    try:
        # Check if model exists in registry
        client = MlflowClient()
        try:
            registered_model = client.get_registered_model(model_registry_name)
            logger.info(f"Model {model_registry_name} already exists in registry")
        except mlflow.exceptions.RestException:
            # Model doesn't exist, create it
            logger.info(f"Creating new registered model: {model_registry_name}")
        
        # Register model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_registry_name,
            description=f"""
            Best performing bonsai species classifier from GitHub Actions CI/CD pipeline.
            
            Model: {best_model_info['model_name']}
            Accuracy: {best_model_info['accuracy']:.4f}
            F1-Score: {best_model_info['f1_score']:.4f}
            
            Trained automatically via:
            - GitHub Actions CI/CD
            - Airflow orchestration
            - MLflow tracking
            
            Configuration: {best_model_info['config']['description']}
            """
        )
        
        # Add tags to the model version
        client = MlflowClient()
        client.set_model_version_tag(
            name=model_registry_name,
            version=model_version.version,
            key="source",
            value="github_actions_cicd"
        )
        
        client.set_model_version_tag(
            name=model_registry_name,
            version=model_version.version,
            key="model_type",
            value=best_model_info['model_name']
        )
        
        client.set_model_version_tag(
            name=model_registry_name,
            version=model_version.version,
            key="accuracy",
            value=str(round(best_model_info['accuracy'], 4))
        )
        
        logger.info(f"✅ Model registered successfully!")
        logger.info(f"   Name: {model_registry_name}")
        logger.info(f"   Version: {model_version.version}")
        logger.info(f"   Accuracy: {best_model_info['accuracy']:.4f}")
        
        return {
            "model_name": model_registry_name,
            "version": model_version.version,
            "run_id": best_model_info['run_id'],
            "accuracy": best_model_info['accuracy']
        }
        
    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Automated model training for bonsai classification")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", default="GitHub-Actions-Bonsai-Training", help="MLflow experiment name")
    parser.add_argument("--model-name", default="GitHub-Actions-Bonsai-Classifier", help="Model registry name")
    parser.add_argument("--no-register", action="store_true", help="Skip model registration")
    
    args = parser.parse_args()
    
    # Configure MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    logger.info(f"🔗 MLflow Tracking URI: {args.mlflow_uri}")
    
    try:
        # Create dataset
        df, feature_names, species_names = create_bonsai_dataset()
        
        # Prepare data
        X = df[feature_names].values
        y = df['species_id'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"📊 Data split: {len(X_train)} train, {len(X_test)} test samples")
        
        # Train models
        best_model_info = train_model_variants(
            X_train, X_test, y_train, y_test, 
            experiment_name=args.experiment_name
        )
        
        # Register best model (if not skipped)
        if not args.no_register and best_model_info:
            registration_info = register_best_model(best_model_info, args.model_name)
            
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"🏆 Best model: {best_model_info['model_name']}")
            logger.info(f"📊 Accuracy: {best_model_info['accuracy']:.4f}")
            logger.info(f"📦 Registered as: {registration_info['model_name']} v{registration_info['version']}")
            
            return registration_info
        else:
            logger.info("🎉 Training pipeline completed successfully!")
            logger.info(f"🏆 Best model: {best_model_info['model_name']}")
            logger.info(f"📊 Accuracy: {best_model_info['accuracy']:.4f}")
            logger.info("⚠️ Model registration skipped")
            
            return best_model_info
            
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
