from fastapi import FastAPI, Request, HTTPException
import mlflow.pyfunc
import mlflow
import numpy as np
import os

app = FastAPI()

# Bonsai species mapping for our plant e-commerce website
BONSAI_SPECIES = {
    0: "Juniper",
    1: "Ficus", 
    2: "Pine",
    3: "Maple"
}

# Care recommendations for each bonsai species
CARE_RECOMMENDATIONS = {
    0: "Hardy evergreen, needs full sun, minimal watering, wire training in fall",
    1: "Prefers bright indirect light, consistent moisture, frequent pruning required", 
    2: "Requires full sun, well-draining soil, candle pinching in spring",
    3: "Needs partial shade, consistent moisture, protection from wind"
}

# Try to load bonsai model from registry, fallback to artifacts volume structure if needed
model = None
model_source = ""

# Set up MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'http://mlflow:5000'
mlflow.set_tracking_uri('http://mlflow:5000')

# Strategy 1: Try to load from Model Registry (preferred)
def load_bonsai_model():
    global model, model_source
    try:
        model = mlflow.pyfunc.load_model("models:/Bonsai-Species-Classifier-Production/Production")
        model_source = "Model Registry (Production)"
        print("✅ Bonsai model loaded from registry: Bonsai-Species-Classifier-Production/Production")
    except Exception as e:
        print(f"⚠️ Could not load from registry: {e}")
        
        # Strategy 2: Try to load latest version from any stage
        try:
            model = mlflow.pyfunc.load_model("models:/Bonsai-Species-Classifier-Production/latest")
            model_source = "Model Registry (Latest)"
            print("✅ Bonsai model loaded from registry: latest version")
        except Exception as e2:
            print(f"⚠️ Could not load latest from registry: {e2}")
            
            # Strategy 3: Try to find the latest model from artifacts volume
            try:
                import glob
                from mlflow.tracking import MlflowClient
                
                client = MlflowClient()
                
                experiment = client.get_experiment_by_name("Bonsai-Species-Classification")
                if experiment:
                    runs = client.search_runs(
                        experiment_ids=[experiment.experiment_id],
                        order_by=["start_time DESC"],
                        max_results=1
                    )
                    
                    if runs:
                        latest_run_id = runs[0].info.run_id
                        model = mlflow.pyfunc.load_model(f"runs:/{latest_run_id}/bonsai_classifier")
                        model_source = f"Latest Run Artifact ({latest_run_id[:8]})"
                        print(f"✅ Bonsai model loaded from run: {latest_run_id}")
                    else:
                        print("❌ No runs found in Bonsai-Species-Classification experiment")
                else:
                    print("❌ Bonsai-Species-Classification experiment not found")
                    
            except Exception as e3:
                print(f"❌ Could not load from run artifacts: {e3}")
                
                # Strategy 4: Direct file system access to artifacts (last resort)
                try:
                    artifact_paths = glob.glob("/artifacts/*/*/artifacts/bonsai_classifier")
                    if artifact_paths:
                        model_path = artifact_paths[0]
                        model = mlflow.pyfunc.load_model(model_path)
                        model_source = f"Direct Artifact Path ({model_path})"
                        print(f"✅ Bonsai model loaded from direct path: {model_path}")
                    else:
                        print("❌ No bonsai_classifier artifacts found in /artifacts")
                        print("💡 Make sure to run the notebook first to create and register a model!")
                except Exception as e4:
                    print(f"❌ Final fallback failed: {e4}")
                    print("💡 Please run the Jupyter notebook to train and register a bonsai model first!")

# Call the model loading function at startup
load_bonsai_model()

@app.get("/")
async def root():
    load_bonsai_model()
    return {
        "message": "🌱 Bonsai Species Classification API for Plant E-commerce", 
        "model_loaded": model is not None,
        "model_source": model_source if model else "No model loaded",
        "species": list(BONSAI_SPECIES.values()),
        "features": ["leaf_length_cm", "leaf_width_cm", "branch_thickness_mm", "height_cm"],
        "mlflow_uri": os.environ.get('MLFLOW_TRACKING_URI', 'Not set')
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy", 
        "model_ready": model is not None,
        "model_source": model_source if model else "No model loaded",
        "mlflow_uri": os.environ.get('MLFLOW_TRACKING_URI', 'Not set'),
        "artifacts_accessible": os.path.exists("/artifacts")
    }

@app.post("/predict")
async def predict(request: Request):
    if model is None:
        raise HTTPException(status_code=503, detail="Bonsai classification model not loaded")
    
    data = await request.json()
    
    # Validate input features
    if "features" not in data:
        raise HTTPException(status_code=400, detail="Missing 'features' field")
    
    features = data["features"]
    if len(features) != 4:
        raise HTTPException(
            status_code=400, 
            detail="Expected 4 features: [leaf_length_cm, leaf_width_cm, branch_thickness_mm, height_cm]"
        )
    
    try:
        # Make prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        prediction_id = int(prediction[0])
        
        # Get species name and care info
        species_name = BONSAI_SPECIES.get(prediction_id, "Unknown")
        care_info = CARE_RECOMMENDATIONS.get(prediction_id, "No care information available")
        
        return {
            "prediction": prediction_id,
            "species": species_name,
            "confidence": "high",  # Could add actual confidence scores
            "care_recommendations": care_info,
            "input_features": {
                "leaf_length_cm": features[0],
                "leaf_width_cm": features[1], 
                "branch_thickness_mm": features[2],
                "height_cm": features[3]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/species")
async def get_species():
    """Get information about all bonsai species we can classify"""
    return {
        "species_info": {
            species_name: {
                "id": species_id,
                "name": species_name,
                "care": CARE_RECOMMENDATIONS[species_id]
            }
            for species_id, species_name in BONSAI_SPECIES.items()
        }
    }
