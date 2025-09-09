"""
LLMOps Prompt Engineering and Fine-tuning Pipeline
Triggered by GitHub Actions, orchestrated by Airflow
Focus: Customer Service Chatbot for Plant Care Assistance
"""

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pandas as pd
import numpy as np
import logging
import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any
import requests
import time

# LLM and prompt engineering imports
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from openai import AzureOpenAI
import litellm

# Evaluation imports
from rouge_score import rouge_scorer
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantCarePromptEngineer:
    """
    LLMOps Pipeline for Plant Care Customer Service
    Focuses on prompt engineering, model evaluation, and deployment
    """
    
    def __init__(self, azure_endpoint: str = None):
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
        self.deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt-4o")
        self.mlflow_experiment = "plant-care-llmops"
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_version=self.azure_api_version,
            azure_endpoint=self.azure_endpoint,
            api_key=self.azure_api_key,
        )
        
    def validate_azure_setup(self):
        """Validate Azure OpenAI configuration"""
        logger.info(f"🤖 Validating Azure OpenAI setup...")
        
        try:
            if not all([self.azure_endpoint, self.azure_api_key, self.deployment_name]):
                logger.error("❌ Missing Azure OpenAI configuration")
                return False
                
            # Test API connection
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            
            if response.choices:
                logger.info(f"✅ Azure OpenAI connection successful")
                return True
            else:
                logger.error(f"❌ Failed to get response from Azure OpenAI")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error validating Azure OpenAI: {str(e)}")
            return False
    
    def create_plant_care_dataset(self) -> pd.DataFrame:
        """Create synthetic plant care customer service dataset"""
        logger.info("📊 Creating plant care customer service dataset...")
        
        # Synthetic customer queries and expected responses
        data = [
            {
                "customer_query": "My plant leaves are turning yellow, what should I do?",
                "category": "disease_diagnosis",
                "expected_response": "Yellow leaves often indicate overwatering or nutrient deficiency. Check soil moisture and reduce watering frequency. Consider fertilizing if soil is depleted.",
                "urgency": "medium"
            },
            {
                "customer_query": "How often should I water my succulent?",
                "category": "watering_advice",
                "expected_response": "Succulents need infrequent watering. Water only when soil is completely dry, typically every 1-2 weeks. Ensure proper drainage.",
                "urgency": "low"
            },
            {
                "customer_query": "My plant has white spots on leaves, is it sick?",
                "category": "disease_diagnosis", 
                "expected_response": "White spots could indicate powdery mildew or pest infestation. Isolate the plant and treat with appropriate fungicide or insecticidal soap.",
                "urgency": "high"
            },
            {
                "customer_query": "What's the best fertilizer for indoor plants?",
                "category": "fertilizer_advice",
                "expected_response": "Use balanced liquid fertilizer (20-20-20) diluted to half strength. Apply monthly during growing season, less in winter.",
                "urgency": "low"
            },
            {
                "customer_query": "My plant isn't growing, help!",
                "category": "growth_issues",
                "expected_response": "Slow growth can be due to insufficient light, poor soil, or wrong season. Check light requirements and consider repotting with fresh soil.",
                "urgency": "medium"
            },
            {
                "customer_query": "Can I propagate my plant? How?",
                "category": "propagation",
                "expected_response": "Most plants can be propagated through stem cuttings. Cut healthy stem, remove lower leaves, place in water or soil. Keep humid and wait for roots.",
                "urgency": "low"
            },
            {
                "customer_query": "My plant is drooping and looks sad!",
                "category": "general_health",
                "expected_response": "Drooping usually indicates watering issues. Check soil moisture - it could be too dry or too wet. Adjust watering accordingly.",
                "urgency": "medium"
            },
            {
                "customer_query": "What temperature is best for houseplants?",
                "category": "environmental_conditions",
                "expected_response": "Most houseplants prefer temperatures between 65-75°F (18-24°C). Avoid cold drafts and sudden temperature changes.",
                "urgency": "low"
            }
        ]
        
        # Expand dataset with variations
        expanded_data = []
        for item in data:
            # Add original
            expanded_data.append(item)
            
            # Add variations
            variations = self._create_query_variations(item)
            expanded_data.extend(variations)
        
        df = pd.DataFrame(expanded_data)
        logger.info(f"✅ Created dataset with {len(df)} samples")
        return df
    
    def _create_query_variations(self, original_item: Dict) -> List[Dict]:
        """Create variations of customer queries for better training data"""
        variations = []
        
        # Simple variations (in real scenario, use data augmentation techniques)
        if "water" in original_item["customer_query"]:
            variations.append({
                **original_item,
                "customer_query": original_item["customer_query"].replace("water", "irrigate")
            })
        
        if "plant" in original_item["customer_query"]:
            variations.append({
                **original_item,
                "customer_query": original_item["customer_query"].replace("plant", "houseplant")
            })
        
        return variations[:2]  # Limit variations
    
    def design_prompts(self) -> Dict[str, PromptTemplate]:
        """Design different prompt templates for plant care assistance"""
        logger.info("📝 Designing prompt templates...")
        
        prompts = {
            "basic_assistant": PromptTemplate(
                input_variables=["query"],
                template="""You are a helpful plant care assistant. Answer the following question about plant care:

Question: {query}

Answer:"""
            ),
            
            "expert_botanist": PromptTemplate(
                input_variables=["query"],
                template="""You are an expert botanist with 20 years of experience in plant care. 
Provide detailed, scientific, yet accessible advice for the following plant care question:

Customer Question: {query}

Expert Response:"""
            ),
            
            "friendly_helper": PromptTemplate(
                input_variables=["query"],
                template="""You are a friendly plant care helper who loves helping people grow healthy plants! 
Be encouraging and provide practical step-by-step advice.

Plant Parent Question: {query}

Friendly Advice:"""
            ),
            
            "structured_diagnostic": PromptTemplate(
                input_variables=["query"],
                template="""You are a plant care diagnostic assistant. Analyze the question and provide:
1. Problem Identification
2. Possible Causes 
3. Recommended Actions
4. Prevention Tips

Question: {query}

Structured Response:"""
            )
        }
        
        logger.info(f"✅ Created {len(prompts)} prompt templates")
        return prompts
    
    def evaluate_prompt_performance(self, prompt_name: str, prompt: PromptTemplate, 
                                  test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate prompt performance using various metrics"""
        logger.info(f"📊 Evaluating prompt: {prompt_name}")
        
        try:
            responses = []
            expected_responses = []
            
            # Generate responses for test samples (limit for demo)
            for idx, row in test_data.head(3).iterrows():  # Limit for educational demo
                try:
                    # Format prompt with query
                    formatted_prompt = prompt.format(query=row['customer_query'])
                    
                    # Call Azure OpenAI
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[{"role": "user", "content": formatted_prompt}],
                        max_tokens=300,
                        temperature=0.7
                    )
                    
                    ai_response = response.choices[0].message.content
                    responses.append(ai_response)
                    expected_responses.append(row['expected_response'])
                    
                    # Add delay to avoid rate limiting
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate response for query {idx}: {str(e)}")
                    responses.append("Error generating response")
                    expected_responses.append(row['expected_response'])
            
            # Calculate metrics
            metrics = self._calculate_response_metrics(responses, expected_responses)
            metrics['prompt_name'] = prompt_name
            
            logger.info(f"✅ Evaluation complete for {prompt_name}")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluating prompt {prompt_name}: {str(e)}")
            return {"prompt_name": prompt_name, "error": str(e)}
    
    def _calculate_response_metrics(self, responses: List[str], 
                                  expected_responses: List[str]) -> Dict[str, float]:
        """Calculate various metrics for response quality"""
        
        metrics = {}
        
        try:
            # ROUGE scores for semantic similarity
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], 
                                            use_stemmer=True)
            
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for response, expected in zip(responses, expected_responses):
                if isinstance(response, str) and isinstance(expected, str):
                    scores = scorer.score(expected, response)
                    rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                    rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                    rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            # Average ROUGE scores
            metrics['rouge1_avg'] = np.mean(rouge_scores['rouge1']) if rouge_scores['rouge1'] else 0
            metrics['rouge2_avg'] = np.mean(rouge_scores['rouge2']) if rouge_scores['rouge2'] else 0
            metrics['rougeL_avg'] = np.mean(rouge_scores['rougeL']) if rouge_scores['rougeL'] else 0
            
            # Response length analysis
            response_lengths = [len(r.split()) for r in responses if isinstance(r, str)]
            metrics['avg_response_length'] = np.mean(response_lengths) if response_lengths else 0
            
            # Response rate (non-error responses)
            valid_responses = [r for r in responses if not r.startswith("Error")]
            metrics['response_rate'] = len(valid_responses) / len(responses) if responses else 0
            
        except Exception as e:
            logger.warning(f"Error calculating metrics: {str(e)}")
            metrics['error'] = str(e)
        
        return metrics
    
    def run_llmops_pipeline(self):
        """Main LLMOps pipeline execution"""
        logger.info("🚀 Starting LLMOps Pipeline for Plant Care Assistant")
        
        # Set up MLflow tracking URI
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Set up MLflow experiment
        try:
            experiment = mlflow.get_experiment_by_name(self.mlflow_experiment)
            if experiment is None:
                mlflow.create_experiment(self.mlflow_experiment)
        except:
            mlflow.create_experiment(self.mlflow_experiment)
        
        mlflow.set_experiment(self.mlflow_experiment)
        
        with mlflow.start_run(run_name=f"llmops_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            # Log pipeline parameters
            mlflow.log_param("deployment_name", self.deployment_name)
            mlflow.log_param("pipeline_type", "prompt_engineering")
            mlflow.log_param("focus_area", "plant_care_customer_service")
            mlflow.log_param("mlflow_uri", mlflow_uri)
            mlflow.log_param("azure_endpoint", self.azure_endpoint)
            
            # 1. Validate Azure OpenAI setup
            model_ready = self.validate_azure_setup()
            mlflow.log_metric("azure_openai_setup_success", 1 if model_ready else 0)
            
            if not model_ready:
                logger.error("❌ Azure OpenAI setup failed")
                return False
            
            # 2. Create dataset
            dataset = self.create_plant_care_dataset()
            mlflow.log_metric("dataset_size", len(dataset))
            
            # Save dataset as artifact
            dataset_path = "plant_care_dataset.csv"
            dataset.to_csv(dataset_path, index=False)
            mlflow.log_artifact(dataset_path, "datasets")
            
            # 3. Design prompts
            prompts = self.design_prompts()
            mlflow.log_metric("num_prompt_templates", len(prompts))
            
            # 4. Evaluate prompts
            best_prompt_name = None
            best_score = 0
            
            for prompt_name, prompt_template in prompts.items():
                logger.info(f"🧪 Testing prompt: {prompt_name}")
                
                if model_ready:
                    metrics = self.evaluate_prompt_performance(
                        prompt_name, prompt_template, dataset
                    )
                else:
                    # Mock metrics for demo when model isn't available
                    metrics = {
                        'prompt_name': prompt_name,
                        'rouge1_avg': np.random.uniform(0.3, 0.8),
                        'rouge2_avg': np.random.uniform(0.2, 0.6),
                        'rougeL_avg': np.random.uniform(0.3, 0.7),
                        'avg_response_length': np.random.uniform(50, 150),
                        'response_rate': np.random.uniform(0.8, 1.0)
                    }
                
                # Log metrics to MLflow
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(f"{prompt_name}_{metric_name}", value)
                
                # Track best performing prompt
                overall_score = metrics.get('rouge1_avg', 0) + metrics.get('rougeL_avg', 0)
                if overall_score > best_score:
                    best_score = overall_score
                    best_prompt_name = prompt_name
            
            # 5. Log best prompt
            mlflow.log_param("best_prompt", best_prompt_name)
            mlflow.log_metric("best_prompt_score", best_score)
            
            # 6. Save best prompt as artifact
            if best_prompt_name:
                best_prompt_dict = {
                    "name": best_prompt_name,
                    "template": prompts[best_prompt_name].template,
                    "input_variables": prompts[best_prompt_name].input_variables,
                    "score": best_score
                }
                
                prompt_path = "/tmp/best_prompt.json"
                with open(prompt_path, 'w') as f:
                    json.dump(best_prompt_dict, f, indent=2)
                
                mlflow.log_artifact(prompt_path, "prompts")
            
            logger.info("✅ LLMOps Pipeline completed successfully")
            logger.info(f"🏆 Best performing prompt: {best_prompt_name} (score: {best_score:.3f})")
            
            return {
                "status": "success",
                "best_prompt": best_prompt_name,
                "best_score": best_score,
                "experiment_name": self.mlflow_experiment
            }

def main():
    """Main function for CLI execution"""
    parser = argparse.ArgumentParser(description='LLMOps Prompt Engineering Pipeline')
    parser.add_argument('--azure-endpoint', 
                       help='Azure OpenAI endpoint URL')
    parser.add_argument('--deployment-name', default='gpt-4o',
                       help='Azure OpenAI deployment name')
    parser.add_argument('--mlflow-uri', default='http://localhost:5000',
                       help='MLflow tracking URI')
    
    args = parser.parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(args.mlflow_uri)
    
    # Initialize pipeline
    pipeline = PlantCarePromptEngineer(azure_endpoint=args.azure_endpoint)
    if args.deployment_name:
        pipeline.deployment_name = args.deployment_name
    
    # Run pipeline
    result = pipeline.run_llmops_pipeline()
    
    print(f"\n🎯 Pipeline Result: {result}")

if __name__ == "__main__":
    main()
