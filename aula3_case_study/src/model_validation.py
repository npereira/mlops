"""
Model Validation Script
Validates model performance and readiness for production deployment
"""

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import logging
import os
import argparse
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def __init__(self, mlflow_uri="http://localhost:5000"):
        self.mlflow_uri = mlflow_uri
        mlflow.set_tracking_uri(mlflow_uri)
        self.client = mlflow.tracking.MlflowClient()
        
    def get_latest_model(self, model_name):
        """Get the latest registered model version"""
        try:
            latest_versions = self.client.get_latest_versions(model_name, stages=["None"])
            if not latest_versions:
                raise ValueError(f"No model versions found for {model_name}")
            
            latest_version = latest_versions[0]
            logger.info(f"📦 Found latest model: {model_name} v{latest_version.version}")
            return latest_version
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest model: {e}")
            raise
    
    def load_model_for_validation(self, model_name, version=None, stage=None):
        """Load model from MLflow Model Registry"""
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            elif stage:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                # Get latest version
                latest_version = self.get_latest_model(model_name)
                model_uri = f"models:/{model_name}/{latest_version.version}"
                version = latest_version.version
            
            logger.info(f"🔄 Loading model from: {model_uri}")
            model = mlflow.sklearn.load_model(model_uri)
            
            return model, version
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def create_validation_dataset(self):
        """Create validation dataset (similar to training but with different seed)"""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        
        logger.info("📊 Creating validation dataset...")
        
        # Generate validation data with different random seed
        X, y = make_classification(
            n_samples=500,
            n_features=4,
            n_classes=4,
            n_informative=4,
            n_redundant=0,
            random_state=123  # Different seed for validation
        )
        
        # Scale features to realistic bonsai measurements
        X[:, 0] = X[:, 0] * 0.5 + 2.0    # leaf_length: 1.5-2.5 cm
        X[:, 1] = X[:, 1] * 0.3 + 1.5    # leaf_width: 1.2-1.8 cm
        X[:, 2] = X[:, 2] * 2.0 + 5.0    # branch_thickness: 3-7 mm
        X[:, 3] = X[:, 3] * 10.0 + 25.0  # height: 15-35 cm
        
        feature_names = ['leaf_length_cm', 'leaf_width_cm', 'branch_thickness_mm', 'height_cm']
        species_names = ['Juniper', 'Ficus', 'Pine', 'Maple']
        
        return X, y, feature_names, species_names
    
    def validate_model_performance(self, model, X_val, y_val, min_accuracy=0.7):
        """Validate model meets minimum performance requirements"""
        logger.info("🎯 Validating model performance...")
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        
        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist()
        }
        
        # Validation checks
        validation_results = {
            "accuracy_check": accuracy >= min_accuracy,
            "precision_check": precision >= 0.7,
            "recall_check": recall >= 0.7,
            "f1_check": f1 >= 0.7
        }
        
        all_checks_passed = all(validation_results.values())
        
        logger.info(f"📊 Performance Metrics:")
        logger.info(f"   Accuracy: {accuracy:.4f} ({'✅' if validation_results['accuracy_check'] else '❌'})")
        logger.info(f"   Precision: {precision:.4f} ({'✅' if validation_results['precision_check'] else '❌'})")
        logger.info(f"   Recall: {recall:.4f} ({'✅' if validation_results['recall_check'] else '❌'})")
        logger.info(f"   F1-Score: {f1:.4f} ({'✅' if validation_results['f1_check'] else '❌'})")
        
        return metrics, validation_results, all_checks_passed
    
    def validate_model_inference(self, model, feature_names):
        """Test model inference with sample data"""
        logger.info("🧪 Testing model inference...")
        
        # Test cases for different bonsai species
        test_cases = [
            {
                "name": "Juniper Bonsai",
                "features": [1.8, 1.2, 4.0, 20.0],
                "expected_species": "Juniper"
            },
            {
                "name": "Ficus Bonsai",
                "features": [2.3, 1.7, 6.0, 28.0],
                "expected_species": "Ficus"
            },
            {
                "name": "Pine Bonsai",
                "features": [2.0, 1.1, 5.5, 30.0],
                "expected_species": "Pine"
            },
            {
                "name": "Maple Bonsai",
                "features": [2.2, 1.8, 5.0, 25.0],
                "expected_species": "Maple"
            }
        ]
        
        species_map = {0: "Juniper", 1: "Ficus", 2: "Pine", 3: "Maple"}
        inference_results = []
        
        for test_case in test_cases:
            try:
                features = np.array([test_case["features"]])
                prediction = model.predict(features)[0]
                predicted_species = species_map[prediction]
                
                # Get prediction probabilities if available
                try:
                    probabilities = model.predict_proba(features)[0]
                    confidence = float(max(probabilities))
                except:
                    confidence = None
                
                result = {
                    "test_case": test_case["name"],
                    "input_features": test_case["features"],
                    "prediction": int(prediction),
                    "predicted_species": predicted_species,
                    "confidence": confidence,
                    "inference_successful": True
                }
                
                inference_results.append(result)
                logger.info(f"✅ {test_case['name']}: {predicted_species} (confidence: {confidence:.3f if confidence else 'N/A'})")
                
            except Exception as e:
                result = {
                    "test_case": test_case["name"],
                    "input_features": test_case["features"],
                    "error": str(e),
                    "inference_successful": False
                }
                inference_results.append(result)
                logger.error(f"❌ {test_case['name']}: Inference failed - {e}")
        
        successful_inferences = sum(1 for r in inference_results if r["inference_successful"])
        inference_success_rate = successful_inferences / len(test_cases)
        
        return inference_results, inference_success_rate >= 0.9
    
    def validate_model_compatibility(self, model):
        """Check model compatibility and requirements"""
        logger.info("🔧 Checking model compatibility...")
        
        compatibility_checks = {
            "sklearn_model": hasattr(model, 'predict'),
            "predict_proba_available": hasattr(model, 'predict_proba'),
            "feature_count": True,  # Will be updated based on actual test
            "serialization": True   # Will be tested
        }
        
        try:
            # Test serialization
            import pickle
            import tempfile
            
            with tempfile.NamedTemporaryFile() as tmp:
                pickle.dump(model, tmp)
                compatibility_checks["serialization"] = True
                logger.info("✅ Model serialization test passed")
                
        except Exception as e:
            compatibility_checks["serialization"] = False
            logger.error(f"❌ Model serialization test failed: {e}")
        
        # Test feature count (expecting 4 features)
        try:
            test_features = np.array([[2.0, 1.5, 5.0, 25.0]])
            model.predict(test_features)
            compatibility_checks["feature_count"] = True
            logger.info("✅ Feature count compatibility test passed")
            
        except Exception as e:
            compatibility_checks["feature_count"] = False
            logger.error(f"❌ Feature count compatibility test failed: {e}")
        
        all_compatible = all(compatibility_checks.values())
        
        return compatibility_checks, all_compatible
    
    def run_full_validation(self, model_name, version=None, stage=None, min_accuracy=0.75):
        """Run complete model validation pipeline"""
        logger.info(f"🚀 Starting full validation for model: {model_name}")
        
        validation_report = {
            "model_name": model_name,
            "validation_timestamp": datetime.now().isoformat(),
            "mlflow_uri": self.mlflow_uri,
            "validation_passed": False,
            "errors": []
        }
        
        try:
            # Load model
            model, model_version = self.load_model_for_validation(model_name, version, stage)
            validation_report["model_version"] = model_version
            
            # Create validation dataset
            X_val, y_val, feature_names, species_names = self.create_validation_dataset()
            validation_report["validation_samples"] = len(X_val)
            
            # Performance validation
            metrics, performance_checks, performance_passed = self.validate_model_performance(
                model, X_val, y_val, min_accuracy
            )
            validation_report["performance_metrics"] = metrics
            validation_report["performance_checks"] = performance_checks
            validation_report["performance_passed"] = performance_passed
            
            # Inference validation
            inference_results, inference_passed = self.validate_model_inference(model, feature_names)
            validation_report["inference_results"] = inference_results
            validation_report["inference_passed"] = inference_passed
            
            # Compatibility validation
            compatibility_checks, compatibility_passed = self.validate_model_compatibility(model)
            validation_report["compatibility_checks"] = compatibility_checks
            validation_report["compatibility_passed"] = compatibility_passed
            
            # Overall validation result
            validation_report["validation_passed"] = (
                performance_passed and 
                inference_passed and 
                compatibility_passed
            )
            
            if validation_report["validation_passed"]:
                logger.info("🎉 Model validation PASSED! Model ready for production.")
            else:
                logger.warning("⚠️ Model validation FAILED! Review issues before production deployment.")
                
            return validation_report
            
        except Exception as e:
            logger.error(f"❌ Validation failed with error: {e}")
            validation_report["errors"].append(str(e))
            validation_report["validation_passed"] = False
            return validation_report

def main():
    parser = argparse.ArgumentParser(description="Model validation for production readiness")
    parser.add_argument("--mlflow-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument("--model-name", required=True, help="Model name in MLflow Model Registry")
    parser.add_argument("--version", help="Specific model version to validate")
    parser.add_argument("--stage", help="Model stage to validate (Production, Staging)")
    parser.add_argument("--min-accuracy", type=float, default=0.75, help="Minimum required accuracy")
    parser.add_argument("--output-file", help="Save validation report to JSON file")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ModelValidator(args.mlflow_uri)
    
    # Run validation
    validation_report = validator.run_full_validation(
        model_name=args.model_name,
        version=args.version,
        stage=args.stage,
        min_accuracy=args.min_accuracy
    )
    
    # Save report if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        logger.info(f"📄 Validation report saved to: {args.output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("MODEL VALIDATION SUMMARY")
    print("="*60)
    print(f"Model: {validation_report['model_name']} v{validation_report.get('model_version', 'N/A')}")
    print(f"Status: {'✅ PASSED' if validation_report['validation_passed'] else '❌ FAILED'}")
    
    if validation_report.get('performance_metrics'):
        metrics = validation_report['performance_metrics']
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if validation_report['errors']:
        print("\nErrors:")
        for error in validation_report['errors']:
            print(f"  - {error}")
    
    print("="*60)
    
    # Exit with appropriate code
    exit(0 if validation_report['validation_passed'] else 1)

if __name__ == "__main__":
    main()
