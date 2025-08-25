import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

if __name__ == "__main__":
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        n_estimators = 100
        clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Log parameters and metrics
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_metric('accuracy', acc)
        
        # Log model and register it
        model_name = "Iris-RandomForest-Best"
        mlflow.sklearn.log_model(
            clf, 
            "model",
            registered_model_name=model_name
        )
        
        print(f'Accuracy: {acc:.2f}')
        print(f'Model registered as: {model_name}')
