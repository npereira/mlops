# MLOps Repository

This repository contains materials and code for MLOps and LLMOps case studies, including infrastructure setup, API development, model training, and prompt engineering workflows.

## Structure

- **aula2_mlop_infra/**
  - Infrastructure and API setup for MLOps.
  - `api/`: Flask API for model serving.
  - `docker/`: Dockerfiles and compose for environment setup.
  - `notebooks/`: Jupyter notebooks for model exploration.
  - `src/`: Scripts for model training.

- **aula3_case_study/**
  - Case study focused on LLMOps and workflow orchestration.
  - `airflow/`: Airflow DAGs, configs, and logs for pipeline automation.
  - `engine/`: (Reserved for pipeline engine code.)
  - `api/`: API and LLM application code.
  - `docker/`: Docker setup for Airflow and API.
  - `notebooks/`: Jupyter notebooks for prompt engineering and MLflow exploration.
  - `src/`: Scripts for prompt engineering, model validation, and training.
  - `tests/`: Unit tests for API and infrastructure.

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

## License
MIT License.
