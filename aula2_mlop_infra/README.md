

# Class II: Infrastructure as Code for MLOps

## Goal
Learn Infrastructure as Code (IaC) principles by deploying a complete MLOps stack using Docker containers. Build a **bonsai species classifier** for a plant website while understanding how to codify, version, and manage ML infrastructure with experiment tracking, model registry, and model serving APIs.

## 🌱 Project Context: Bonsai Species Classifier
We'll build a machine learning model to classify different bonsai species for an e-commerce plant website. The model will help customers identify bonsai types from photos and provide care recommendations.

## Class Structure

### Prerequisites
Students only need:
- **Docker** and **Docker Compose** installed
- A web browser
- No Python, MLflow, or other tools needed locally!

### Quick Start
1. Clone the repository
2. Navigate to `aulas/aula2_mlop_infra/docker`
3. Run: `docker compose up -d`

4. Open your browser to:
   - **JupyterLab**: http://localhost:8888 (main workspace - train bonsai classifier)
   - **MLflow UI**: http://localhost:5000 (experiment tracking - compare bonsai models)
   - **API**: http://localhost:8080 (model serving - bonsai species prediction)

### 1. Infrastructure as Code Foundations (45 min)
#### Theory & Concepts (20 min)
- **IaC Principles**: Why codify infrastructure? Version control, reproducibility, scalability
- **Docker Fundamentals**: Containers vs. VMs, images, networking, volumes
- **Container Orchestration**: Docker Compose for multi-service applications
- **MLOps Infrastructure Patterns**: Common architectures and best practices

#### Live Demo & Setup (25 min)
- **Environment Setup**: Deploy the entire MLOps stack with `docker compose up -d`
- **Service Discovery**: Explore running containers, networks, and volumes
- **Health Checks**: Verify all services are operational
- **Troubleshooting**: Common startup issues and debugging techniques

### 2. Hands-on Bonsai Classification & Infrastructure Deep Dive (90 min)
Students work through the notebook (`bonsai_classifier_mlflow.ipynb`) while mastering containerized infrastructure:

#### Infrastructure Components & Experiment Tracking (45 min)
- **Docker Compose Analysis**: Dissect the YAML configuration line by line
- **Service Networking**: How containers communicate (DNS, ports, networks)
- **Volume Management**: Data persistence strategies for ML workflows
- **Bonsai Dataset**: Create and explore realistic plant classification data
- **MLflow Integration**: Experiment tracking in containerized environments
- **Hands-on Practice**: Train multiple bonsai classifiers with different hyperparameters

#### Model Registry & Advanced Infrastructure (45 min)
- **Model Registry Deep Dive**: Centralized model storage and versioning
- **Model Lifecycle Management**: Staging → Production promotion workflows
- **Container Resource Management**: CPU, memory, and storage considerations
- **Infrastructure Monitoring**: Container health, logs, and performance metrics
- **Scaling Strategies**: How to scale services horizontally and vertically
- **Real-world Scenarios**: Production deployment considerations

### 3. API Development & Production Readiness (75 min)
#### API Integration & Testing (35 min)
- **Container-to-Container Communication**: JupyterLab → MLflow → API integration
- **API Development**: Build and test the bonsai species prediction endpoint
- **End-to-end Testing**: Realistic bonsai measurements and species predictions
- **Error Handling**: Robust API responses and fallback mechanisms

#### Infrastructure Operations & Production Readiness (40 min)
- **Infrastructure Validation**: Complete stack verification and health checks
- **Configuration Management**: Environment variables, secrets, and configuration files
- **Backup & Recovery**: Data persistence and disaster recovery strategies
- **Security Considerations**: Container security, network policies, and access control
- **Performance Optimization**: Resource tuning and bottleneck identification
- **Deployment Strategies**: Blue-green, rolling updates, and rollback procedures
- **Preview Next Class**: "We'll automate this entire infrastructure with CI/CD pipelines!"

## 🛠️ Extended Troubleshooting & Operations Guide

### Infrastructure Debugging

**Container Orchestration Issues:**
```bash
# Complete infrastructure health check
docker compose ps
docker compose logs --tail=100

# Resource utilization monitoring
docker stats --no-stream
docker system df

# Network connectivity testing
docker compose exec jupyter curl http://mlflow:5000/health
```

**Windows CMD equivalents:**
```cmd
REM Complete infrastructure health check
docker compose ps
docker compose logs --tail=100

REM Resource utilization monitoring
docker stats --no-stream
docker system df

REM Network connectivity testing
docker compose exec jupyter curl http://mlflow:5000/health
```

**MLflow UI not loading?**
```bash
# Comprehensive MLflow debugging
docker compose logs mlflow
docker compose exec mlflow ls -la /mlflow
docker compose exec mlflow netstat -tlnp

# Port conflict resolution
netstat -ano | findstr :5000
```

**Windows CMD equivalents:**
```cmd
REM Comprehensive MLflow debugging
docker compose logs mlflow
docker compose exec mlflow ls -la /mlflow
docker compose exec mlflow netstat -tlnp

REM Port conflict resolution
netstat -ano | findstr :5000
```

**JupyterLab connection issues?**
```bash
# Access token and configuration
docker compose logs jupyter | grep token
docker compose exec jupyter jupyter lab list

# Custom configuration troubleshooting
docker compose exec jupyter cat /opt/conda/etc/jupyter/jupyter_lab_config.py
```

**Windows CMD equivalents:**
```cmd
REM Access token and configuration
docker compose logs jupyter | findstr token
docker compose exec jupyter jupyter lab list

REM Custom configuration troubleshooting
docker compose exec jupyter type /opt/conda/etc/jupyter/jupyter_lab_config.py
```

**Model Registry errors in notebook?**
- Verify MLflow backend store configuration in docker-compose.yml
- Check if Model Registry is enabled (depends on MLflow setup)
- Test Model Registry API manually: `curl http://localhost:5000/api/2.0/mlflow/registered-models/list`
- Ensure bonsai dataset variables are properly defined across cells
- Review MLflow artifact store permissions and paths

**API connection failed?**
```bash
# Comprehensive API debugging
docker compose logs api

# Test complete prediction pipeline
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [2.1, 1.8, 5.5, 25.3]}' | jq .

# Model loading verification
docker compose exec api python -c "import mlflow; print(mlflow.__version__)"
```

**Windows CMD equivalents:**
```cmd
REM Comprehensive API debugging
docker compose logs api

REM Test complete prediction pipeline (using PowerShell for better JSON handling)
powershell -Command "Invoke-RestMethod -Uri 'http://localhost:8080/predict' -Method Post -ContentType 'application/json' -Body '{\"features\": [2.1, 1.8, 5.5, 25.3]}'"

REM Alternative using curl for Windows
curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"features\": [2.1, 1.8, 5.5, 25.3]}"

REM Model loading verification
docker compose exec api python -c "import mlflow; print(mlflow.__version__)"
```

### Advanced Container Management
```bash
# Service scaling and management
docker compose up -d --scale jupyter=2
docker compose restart mlflow
docker compose down && docker compose up -d

# Volume and data management
docker volume ls
docker volume inspect aula2_mlop_infra_mlflow_data

# Container resource monitoring
docker compose top
docker compose exec mlflow top
docker compose exec jupyter df -h

# Network troubleshooting
docker network ls
docker network inspect aula2_mlop_infra_default
```

**Windows CMD equivalents:**
```cmd
REM Service scaling and management
docker compose up -d --scale jupyter=2
docker compose restart mlflow
docker compose down && docker compose up -d

REM Volume and data management
docker volume ls
docker volume inspect aula2_mlop_infra_mlflow_data

REM Container resource monitoring
docker compose top
docker compose exec mlflow top
docker compose exec jupyter df -h

REM Network troubleshooting
docker network ls
docker network inspect aula2_mlop_infra_default
```

### Performance Optimization
```bash
# Resource allocation tuning (add to docker-compose.yml)
# services:
#   mlflow:
#     deploy:
#       resources:
#         limits:
#           memory: 1G
#           cpus: '0.5'

# Disk space management
docker system prune -a
docker volume prune
```

**Windows CMD equivalents:**
```cmd
REM Resource allocation tuning (add to docker-compose.yml)
REM services:
REM   mlflow:
REM     deploy:
REM       resources:
REM         limits:
REM           memory: 1G
REM           cpus: '0.5'

REM Disk space management
docker system prune -a
docker volume prune
```

## 📁 Extended Project Structure
```
aula2_mlop_infra/
├── notebooks/
│   ├── bonsai_classifier_mlflow.ipynb     # Main bonsai classification workshop
│   ├── infrastructure_exploration.ipynb   # Docker & networking deep dive
│   └── advanced_mlflow_features.ipynb     # Model registry & lifecycle management
├── data/
│   ├── bonsai_dataset.csv                 # Primary bonsai species dataset
│   ├── bonsai_images/                     # Sample bonsai photos (future extension)
│   └── plant_care_recommendations.json    # Species-specific care data
├── src/
│   ├── train_bonsai_model.py              # Automated bonsai model training
│   ├── model_validation.py                # Model performance validation
│   ├── data_preprocessing.py              # Bonsai data preparation pipeline
│   └── infrastructure_health_check.py     # Container monitoring utilities
├── docker/
│   ├── Dockerfile.api                     # API service container
│   ├── Dockerfile.mlflow                  # Custom MLflow container
│   ├── requirements.txt                   # Python dependencies
│   ├── docker-compose.yml                 # Complete infrastructure definition
│   ├── docker-compose.override.yml        # Development overrides
│   └── .env.template                      # Environment configuration template
├── api/
│   ├── app.py                             # Bonsai species prediction API
│   ├── model_loader.py                    # MLflow model loading utilities
│   ├── validation.py                      # Input validation and error handling
│   └── health_checks.py                   # API health and readiness endpoints
├── config/
│   ├── mlflow.conf                        # MLflow server configuration
│   ├── jupyter_config.py                  # JupyterLab customization
│   └── logging.conf                       # Centralized logging configuration
├── scripts/
│   ├── setup_environment.sh               # Automated environment setup
│   ├── backup_data.sh                     # Data backup procedures
│   └── performance_monitoring.sh          # Infrastructure monitoring
├── tests/
│   ├── test_bonsai_classifier.py          # Model testing suite
│   ├── test_api_endpoints.py              # API integration tests
│   └── test_infrastructure.py             # Container health tests
└── README.md                              # This comprehensive guide
```

## 🌳 Advanced Bonsai Species Classification Details

### Extended Dataset Features
- **Leaf Length** (cm): Average leaf measurement across seasonal variations
- **Leaf Width** (cm): Average leaf width with seasonal adjustments
- **Branch Thickness** (mm): Primary trunk diameter at base
- **Height** (cm): Overall plant height from soil to apex
- **Age Estimation** (years): Estimated bonsai age for maturity classification
- **Seasonal Color Index**: Color variation score across seasons (0-10)
- **Root System Type**: Fibrous, tap root, or aerial root classification
- **Bark Texture Score**: Surface roughness and pattern complexity (0-10)

### Extended Target Classes & Care Requirements
- **Juniper Bonsai** (0): 
  - **Characteristics**: Hardy evergreen, needle-like foliage, drought resistant
  - **Care**: Full sun, minimal watering, wire training in fall
  - **Business Value**: Low-maintenance option for beginners

- **Ficus Bonsai** (1): 
  - **Characteristics**: Broad leaves, aerial roots, rapid growth
  - **Care**: Bright indirect light, consistent moisture, frequent pruning
  - **Business Value**: Indoor-friendly, dramatic visual appeal

- **Pine Bonsai** (2): 
  - **Characteristics**: Long needles, distinctive candles, slow growth
  - **Care**: Full sun, well-draining soil, candle pinching in spring
  - **Business Value**: Traditional aesthetic, collector's choice

- **Maple Bonsai** (3): 
  - **Characteristics**: Lobed leaves, seasonal color changes, delicate branching
  - **Care**: Partial shade, consistent moisture, protection from wind
  - **Business Value**: Spectacular autumn display, premium pricing

### Enhanced Business Value Propositions
- **Customer Experience**: 
  - Instant species identification with 95%+ accuracy
  - Personalized care recommendations based on customer location/climate
  - Seasonal care calendar generation
  - Compatibility assessment with customer's existing collection

- **Inventory Management**: 
  - Automated plant categorization and pricing optimization
  - Seasonal demand forecasting based on species characteristics
  - Quality control verification against supplier descriptions
  - Cross-selling recommendations based on species compatibility

- **Operational Efficiency**:
  - Staff training reduction through automated identification
  - Reduced customer service inquiries via self-service tools
  - Dynamic pricing based on rarity and care complexity
  - Supply chain optimization through demand prediction
