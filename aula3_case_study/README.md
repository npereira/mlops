# 🌱🤖 Aula 3: LLMOps and Prompt Engineering with Airflow

## 📚 Learning Objectives

This class introduces students to **LLMOps (Large Language Model Operations)** and **Prompt Engineering** while maintaining focus on MLOps/AIOps principles. Students will learn to:

- Design and evaluate prompt templates for LLM applications
- Implement LLMOps pipelines using Airflow orchestration
- Deploy and monitor LLM-powered services locally (no cloud required)
- Apply MLOps principles to LLM development and deployment
- Use Azure OpenAI for cloud-based LLM deployment and experimentation

## 🎯 Project Focus: Plant Care Assistant + BonsAI Chat Bot

**Main Scenario**: Build an AI-powered customer service assistant for a plant care company that helps customers diagnose plant problems and get care advice.

**Special Feature**: **🌿 BonsAI Chat Bot** - A specialized interactive web-based bonsai care expert that only answers questions about bonsai plants.

**Why This Context?**
- **Relatable**: Everyone can understand plant care questions
- **Practical**: Demonstrates real-world LLM applications
- **Educational**: Clear evaluation criteria (helpful vs unhelpful responses)
- **Engaging**: Students can test with their own plant questions
- **Specialized**: BonsAI demonstrates domain-specific AI assistants

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GitHub        │    │    Airflow       │    │                 │
│   Actions       │────│   Orchestrator   │────│   Azure OpenAI  │
│   (CI/CD)       │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│    MLflow       │    │   Prompt Eng.    │    │   Flask API     │
│   Tracking      │────│   Pipeline       │────│   Chat Service  │
│   & Registry    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Technology Stack

### **Core LLMOps Tools**
- **Azure OpenAI**: Cloud-based LLM deployment with enterprise features
- **LangChain**: LLM application framework
- **MLflow**: Experiment tracking for prompt engineering
- **Apache Airflow**: LLMOps pipeline orchestration

### **Infrastructure**
- **Docker Compose**: Multi-service orchestration
- **PostgreSQL**: Airflow metadata database
- **Flask**: API framework for LLM service

### **Evaluation & Monitoring**
- **ROUGE**: Text similarity metrics
- **Custom metrics**: Response length, relevance
- **A/B testing**: Compare prompt templates

## 📁 Project Structure

```
aulas/aula3_case_study/
├── 🐳 docker/
│   ├── docker-compose.yml       # Multi-service orchestration
│   ├── Dockerfile.api          # LLM API service
│   └── requirements.txt        # LLMOps dependencies
├── 🌊 airflow/
│   ├── dags/
│   │   └── llmops_plant_care.py    # LLMOps pipeline
│   ├── logs/                   # Airflow logs
│   └── plugins/                # Custom operators
├── 🧠 src/
│   └── llm_prompt_engineering.py   # Core LLMOps pipeline
├── 🚀 api/
│   └── llm_app.py              # Plant care chat API
├── 🔄 .github/
│   └── workflows/
│       └── llmops-ci-cd.yml    # GitHub Actions pipeline
├── 🧪 tests/
│   ├── test_prompt_engineering.py  # Prompt quality tests
│   └── test_llm_api.py         # API integration tests
└── 📖 README.md                # This file
```

## 🚀 Quick Start

### 1. **Start the LLMOps Infrastructure**

```bash
# Navigate to docker directory
cd aulas/aula3_case_study/docker

# Start all services
docker-compose up -d

# Check services status
docker-compose ps
```

### 2. **Access the Services**

| Service | URL | Purpose |
|---------|-----|---------|
| **Airflow** | http://localhost:8080 | Pipeline orchestration |
| **MLflow** | http://localhost:5000 | Experiment tracking |
| **🌿 BonsAI Chat** | http://localhost:3000 | Interactive bonsai expert |
| **Jupyter Lab** | http://localhost:8888 | Notebook development |

**Default Credentials:**
- Airflow: `admin` / `admin`

### 2.1. **🌿 BonsAI Chat Bot - Interactive Experience**

The main application is the specialized bonsai care expert with a beautiful web interface:

```bash
# Start all services including BonsAI
docker-compose up -d

# Or start BonsAI standalone
cd api
python bonsai_app.py
```

**Features:**
- 🌿 Interactive web chat interface
- 🎯 Specialized only in bonsai care
- 🔄 Multiple conversation modes (basic, diagnostic, emergency)
- 🎨 Beautiful responsive UI
- 📚 Quick-start questions

**Example Questions for BonsAI:**
- "How often should I water my Juniper bonsai?"
- "My bonsai leaves are yellowing, help!"
- "What soil mix is best for Ficus bonsai?"

### 3. **Test the Plant Care Assistant**

```bash
# Test the chat API
curl -X POST http://localhost:8081/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "My plant leaves are turning yellow, what should I do?"}'

# Get demo queries
curl http://localhost:8081/demo
```

## 🎓 Learning Activities

### **Activity 1: Prompt Engineering Fundamentals**

**Objective**: Understand how different prompts affect LLM responses

1. **Explore Prompt Templates** (`src/llm_prompt_engineering.py`):
   - `basic_assistant`: Simple, direct responses
   - `expert_botanist`: Scientific, detailed advice
   - `friendly_helper`: Encouraging, step-by-step guidance
   - `structured_diagnostic`: Organized problem-solving approach

2. **Run Prompt Comparison**:
```bash
# Trigger the LLMOps pipeline
docker exec -it airflow-scheduler airflow dags trigger llmops_plant_care_pipeline
```

3. **Analyze Results** in MLflow:
   - Compare ROUGE scores between prompts
   - Evaluate response lengths and quality
   - Identify the best-performing template

### **Activity 2: LLMOps Pipeline Development**

**Objective**: Build end-to-end LLM deployment pipeline

1. **Modify Prompt Templates**: Add your own creative prompt
2. **Run Evaluation Pipeline**: Test against plant care dataset
3. **Deploy Best Model**: Automatically update the API service
4. **Monitor Performance**: Track metrics in MLflow

### **Activity 3: GitHub Actions Integration**

**Objective**: Implement CI/CD for LLM applications

1. **Trigger Pipeline**: Push changes to GitHub
2. **Automated Testing**: Run prompt quality tests
3. **Deployment**: Auto-deploy best performing prompts
4. **Monitoring**: Set up alerts for performance degradation

## 🧪 Hands-On Exercises

### **Exercise 1: Design Your Own Prompt**

Create a new prompt template that addresses specific plant care scenarios:

```python
"crisis_manager": PromptTemplate(
    input_variables=["query"],
    template="""You are an emergency plant care specialist. 
    The user has an urgent plant problem that needs immediate attention.
    
    URGENT PLANT ISSUE: {query}
    
    IMMEDIATE ACTION PLAN:"""
)
```

### **Exercise 2: Implement Custom Evaluation Metrics**

Add domain-specific evaluation metrics:

```python
def evaluate_plant_advice_quality(response: str) -> float:
    # Check for actionable advice
    action_words = ["water", "fertilize", "repot", "prune", "move"]
    has_action = any(word in response.lower() for word in action_words)
    
    # Check for specificity
    specific_terms = ["once", "twice", "daily", "weekly", "cm", "inches"]
    has_specificity = any(term in response.lower() for term in specific_terms)
    
    return (has_action + has_specificity) / 2
```

### **Exercise 3: A/B Test Prompt Performance**

Design an A/B testing framework:

1. Split test queries between two prompt templates
2. Collect user feedback ratings
3. Calculate statistical significance
4. Auto-deploy the winning prompt

## 📊 Evaluation Metrics

### **Automated Metrics**
- **ROUGE-1/2/L**: Semantic similarity to reference responses
- **Response Length**: Appropriate detail level
- **Response Rate**: Successful generation percentage
- **Latency**: Time to generate responses

### **Human Evaluation** (Simulated)
- **Helpfulness**: Does the advice solve the problem?
- **Accuracy**: Is the plant care information correct?
- **Clarity**: Is the response easy to understand?
- **Actionability**: Can the user follow the advice?

## 🔄 MLOps Principles Applied to LLMs

### **Version Control**
- **Prompt Templates**: Track changes in MLflow
- **Model Versions**: Manage different LLM configurations
- **Data Versioning**: Track evaluation datasets

### **Continuous Integration**
- **Automated Testing**: Quality checks for new prompts
- **Performance Validation**: Regression testing
- **Deployment Gates**: Quality thresholds

### **Monitoring & Observability**
- **Response Quality**: Track evaluation metrics
- **User Satisfaction**: Feedback collection
- **Performance Monitoring**: Latency and throughput
- **A/B Testing**: Compare prompt variants

## 🌟 Advanced Topics

### **Prompt Optimization Techniques**
- **Few-shot Learning**: Examples in prompts
- **Chain-of-Thought**: Step-by-step reasoning
- **Temperature Tuning**: Creativity vs consistency
- **Token Optimization**: Efficiency improvements

### **Production Considerations**
- **Caching**: Reduce redundant LLM calls
- **Rate Limiting**: Manage API usage
- **Fallback Strategies**: Handle model failures
- **Content Filtering**: Safety and appropriateness

## 🎯 Learning Outcomes

By completing this class, students will:

✅ **Understand LLMOps fundamentals** and how they differ from traditional MLOps
✅ **Design effective prompts** for specific use cases
✅ **Implement evaluation frameworks** for LLM applications  
✅ **Build deployment pipelines** for LLM services
✅ **Apply MLOps principles** to LLM development lifecycle
✅ **Use local LLM tools** without requiring cloud services

## 📚 Additional Resources

- **Azure OpenAI Documentation**: https://docs.microsoft.com/en-us/azure/cognitive-services/openai/
- **LangChain Guide**: https://python.langchain.com/docs/
- **Prompt Engineering Guide**: https://www.promptingguide.ai/
- **MLflow for LLMs**: https://mlflow.org/docs/latest/llms/index.html

## 🆘 Troubleshooting

### **Common Issues**

1. **Azure OpenAI Configuration Issues**:
   ```bash
   # Ensure environment variables are set correctly
   export AZURE_OPENAI_ENDPOINT="your_endpoint"
   export AZURE_OPENAI_API_KEY="your_key"
   export AZURE_DEPLOYMENT_NAME="your_deployment"
   export OPENAI_API_VERSION="2024-02-15-preview"
   ```

2. **API Not Responding**:
   ```bash
   # Check service health
   curl http://localhost:8081/health
   ```

3. **Airflow DAG Not Showing**:
   ```bash
   # Refresh DAGs
   docker exec airflow-scheduler airflow dags list-import-errors
   ```

---

## 🎉 Ready to Build AI That Helps Plants Grow? 

Start your LLMOps journey by exploring prompt engineering for plant care assistance! 🌱✨
