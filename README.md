# MLOps Repository

This repository contains materials and code for MLOps and LLMOps case studies, including infrastructure setup, API development, model training, and prompt engineering workflows.

## Main Folders

- **aula2_mlop_infra/**
  - MLOps infrastructure, API, and model training workflows
  - `api/`: Flask API for model serving
  - `docker/`: Dockerfiles, Compose, and requirements
  - `notebooks/`: Jupyter notebooks for ML experiments
  - `src/`: Model training scripts

- **aula3_case_study/**
  - LLMOps case study and workflow orchestration
  - `airflow/`: DAGs, configs, and logs for Airflow pipelines
  - `api/`: API and LLM application code
  - `docker/`: Docker setup for Airflow and API
  - `notebooks/`: Jupyter notebooks for prompt engineering and MLflow
  - `src/`: Prompt engineering, model validation, and training scripts
  - `tests/`: Unit tests for API and infrastructure

## Automation & CI/CD

- **GitHub Actions**: Automated checks for clean Jupyter notebooks, code quality, and reproducibility
  - Ensures notebooks are free of outputs and execution counts before merging
- **Dependabot**: Automated dependency updates for Python and Docker
  - Keeps dependencies secure and up-to-date

## Getting Started

1. Clone the repository:
   ```sh
   git clone https://github.com/luismaiaDEVSCOPE/mlops.git
   ```
2. Review the README files in each folder for specific setup instructions.
3. Use Docker Compose files for environment setup as needed.

## Requirements
- Python 3.8+
- Docker & Docker Compose

## Useful Links
- MLflow UI: http://localhost:5001
- JupyterLab: http://localhost:8888
- API: http://localhost:8080

## License
MIT License.
