"""
LLMOps Pipeline DAG
Orchestrates prompt engineering, model evaluation, and deployment
Focus: Plant Care Customer Service Assistant
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
import requests
import logging
import json
import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'llmops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'llmops_plant_care_pipeline',
    default_args=default_args,
    description='LLMOps pipeline for plant care assistant',
    schedule=timedelta(hours=6),  # Run every 6 hours
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['llmops', 'prompt-engineering', 'plant-care', 'education']
)

def check_azure_openai_health(**context):
    """Check if Azure OpenAI service is healthy and accessible"""
    logger.info("🔍 Checking Azure OpenAI service health...")
    
    try:
        from openai import AzureOpenAI
        import os
        
        # Get Azure OpenAI configuration
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        azure_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        azure_api_version = os.getenv('OPENAI_API_VERSION', '2024-02-15-preview')
        deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME', 'gpt-4o')
        
        if not all([azure_endpoint, azure_api_key]):
            logger.error("❌ Missing Azure OpenAI configuration")
            return False
            
        # Initialize client and test connection
        client = AzureOpenAI(
            api_version=azure_api_version,
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
        )
        
        # Test with a simple request
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        
        if response.choices:
            logger.info(f"✅ Azure OpenAI is healthy. Model: {deployment_name}")
            return True
        else:
            logger.error(f"❌ No response from Azure OpenAI")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking Azure OpenAI health: {str(e)}")
        return False

def check_mlflow_connection(**context):
    """Verify MLflow tracking server is accessible"""
    logger.info("🔍 Checking MLflow connection...")
    
    try:
        # Set MLflow tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Create MLflow client
        client = MlflowClient()
        
        # Try to list experiments to test connection
        experiments = client.search_experiments()
        logger.info(f"✅ MLflow is accessible. Found {len(experiments)} experiments")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error checking MLflow: {str(e)}")
        return False

def prepare_training_environment(**context):
    """Prepare the environment for LLM training"""
    logger.info("🔧 Preparing training environment...")
    
    # Simulate environment preparation
    # In real scenario: download datasets, prepare data splits, etc.
    
    prep_status = {
        "dataset_ready": True,
        "prompt_templates_ready": True,
        "evaluation_metrics_ready": True,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info("✅ Training environment prepared")
    return prep_status

def run_prompt_engineering_pipeline(**context):
    """Execute the main prompt engineering pipeline"""
    logger.info("🚀 Running prompt engineering pipeline...")
    
    try:
        # Run the LLMOps script
        import subprocess
        import os
        
        cmd = [
            "python", "/opt/airflow/src/llm_prompt_engineering.py",
            "--mlflow-uri", "http://mlflow:5000"
        ]
        
        # Add Azure OpenAI endpoint if configured
        azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        if azure_endpoint:
            cmd.extend(["--azure-endpoint", azure_endpoint])
            
        deployment_name = os.getenv('AZURE_DEPLOYMENT_NAME')
        if deployment_name:
            cmd.extend(["--deployment-name", deployment_name])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 minutes timeout
        
        if result.returncode == 0:
            logger.info("✅ Prompt engineering pipeline completed successfully")
            logger.info(f"Output: {result.stdout}")
            return {"status": "success", "output": result.stdout}
        else:
            logger.error(f"❌ Pipeline failed: {result.stderr}")
            raise Exception(f"Pipeline failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Pipeline timed out")
        raise Exception("Pipeline execution timed out")
    except Exception as e:
        logger.error(f"❌ Error running pipeline: {str(e)}")
        raise

def evaluate_model_performance(**context):
    """Evaluate the performance of prompt templates"""
    logger.info("📊 Evaluating model performance...")
    
    try:
        # Set MLflow tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Create MLflow client
        client = MlflowClient()
        
        # Get or create experiment
        experiment_name = "plant-care-llmops"
        try:
            experiment = client.get_experiment_by_name(experiment_name)
            experiment_id = experiment.experiment_id
        except mlflow.exceptions.MlflowException:
            logger.warning(f"Experiment '{experiment_name}' not found, creating new one")
            experiment_id = client.create_experiment(experiment_name)
        
        # Get latest run from experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs:
            latest_run = runs[0]
            metrics = latest_run.data.metrics
            
            # Extract performance metrics
            performance = {
                "best_prompt_score": metrics.get("best_prompt_score", 0),
                "run_id": latest_run.info.run_id,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Performance evaluation complete: {performance}")
            return performance
        
        # Fallback evaluation
        logger.warning("⚠️ Using fallback evaluation - no runs found")
        return {
            "status": "fallback",
            "score": 0.5,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error evaluating performance: {str(e)}")
        raise

def deploy_best_model(**context):
    """Deploy the best performing model/prompt"""
    logger.info("🚀 Deploying best model...")
    
    # In production, this would:
    # 1. Update the production API configuration
    # 2. Restart the API service with new prompt
    # 3. Run smoke tests
    
    try:
        # Check if API is healthy
        api_response = requests.get("http://api:8080/health", timeout=10)
        
        if api_response.status_code == 200:
            logger.info("✅ API is healthy, deployment successful")
            return {"status": "deployed", "api_health": "healthy"}
        else:
            logger.warning("⚠️ API health check failed")
            return {"status": "deployed", "api_health": "warning"}
            
    except Exception as e:
        logger.error(f"❌ Deployment check failed: {str(e)}")
        return {"status": "deployed", "api_health": "unknown"}

def run_quality_tests(**context):
    """Run quality tests on the deployed model"""
    logger.info("🧪 Running quality tests...")
    
    test_queries = [
        "My plant leaves are turning yellow, what should I do?",
        "How often should I water my succulent?",
        "What's the best fertilizer for indoor plants?"
    ]
    
    test_results = []
    
    for query in test_queries:
        try:
            response = requests.post(
                "http://api:8080/chat",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code == 200:
                result_data = response.json()
                test_results.append({
                    "query": query,
                    "status": "success",
                    "response_length": len(result_data.get("response", "")),
                })
                logger.info(f"✅ Test passed for query: {query[:30]}...")
            else:
                test_results.append({
                    "query": query,
                    "status": "failed",
                    "error": f"HTTP {response.status_code}"
                })
                logger.warning(f"⚠️ Test failed for query: {query[:30]}...")
                
        except Exception as e:
            test_results.append({
                "query": query,
                "status": "error",
                "error": str(e)
            })
            logger.error(f"❌ Test error for query: {query[:30]}... - {str(e)}")
    
    # Calculate success rate
    successful_tests = sum(1 for result in test_results if result["status"] == "success")
    success_rate = successful_tests / len(test_results) if test_results else 0
    
    logger.info(f"🎯 Quality tests completed. Success rate: {success_rate:.2%}")
    
    return {
        "success_rate": success_rate,
        "total_tests": len(test_results),
        "successful_tests": successful_tests,
        "test_results": test_results
    }

def send_notification(**context):
    """Send pipeline completion notification"""
    logger.info("📢 Sending pipeline completion notification...")
    
    # Get results from previous tasks
    task_instance = context['task_instance']
    
    try:
        # Get performance results
        performance = task_instance.xcom_pull(task_ids='evaluate_model_performance')
        quality_results = task_instance.xcom_pull(task_ids='run_quality_tests')
        
        notification = {
            "pipeline": "LLMOps Plant Care Assistant",
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "performance_score": performance.get("best_prompt_score", 0) if performance else 0,
            "quality_success_rate": quality_results.get("success_rate", 0) if quality_results else 0,
            "message": "LLMOps pipeline completed successfully! 🌱🤖"
        }
        
        logger.info(f"✅ Notification sent: {notification}")
        
        # In production, send to Slack, email, etc.
        return notification
        
    except Exception as e:
        logger.error(f"❌ Error sending notification: {str(e)}")
        return {"status": "error", "message": str(e)}

# Define tasks
start_task = EmptyOperator(
    task_id='start_llmops_pipeline',
    dag=dag
)

check_azure_openai_task = PythonOperator(
    task_id='check_azure_openai_health',
    python_callable=check_azure_openai_health,
    dag=dag
)

check_mlflow_task = PythonOperator(
    task_id='check_mlflow_connection',
    python_callable=check_mlflow_connection,
    dag=dag
)

prepare_env_task = PythonOperator(
    task_id='prepare_training_environment',
    python_callable=prepare_training_environment,
    dag=dag
)

run_pipeline_task = PythonOperator(
    task_id='run_prompt_engineering_pipeline',
    python_callable=run_prompt_engineering_pipeline,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_model_performance',
    python_callable=evaluate_model_performance,
    dag=dag
)

deploy_task = PythonOperator(
    task_id='deploy_best_model',
    python_callable=deploy_best_model,
    dag=dag
)

quality_test_task = PythonOperator(
    task_id='run_quality_tests',
    python_callable=run_quality_tests,
    dag=dag
)

notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

end_task = EmptyOperator(
    task_id='end_llmops_pipeline',
    dag=dag
)

# Define task dependencies
start_task >> [check_azure_openai_task, check_mlflow_task]
[check_azure_openai_task, check_mlflow_task] >> prepare_env_task
prepare_env_task >> run_pipeline_task
run_pipeline_task >> evaluate_task
evaluate_task >> deploy_task
deploy_task >> quality_test_task
quality_test_task >> notification_task
notification_task >> end_task
