# 🌿 BonsAI Chat Bot

BonsAI is a specialized bonsai care expert assistant built with MLflow and Azure OpenAI. It provides interactive web-based chat interface focused exclusively on bonsai plant care.

## Features

- **Specialized Knowledge**: Expert advice exclusively for bonsai care
- **Interactive Web UI**: Beautiful, responsive chat interface
- **Multiple Prompt Modes**: Different conversation styles for various needs
- **MLflow Integration**: Experiment tracking and prompt management
- **Azure OpenAI Powered**: Advanced language model capabilities

## Quick Start

### Run with Docker Compose (Recommended)

```bash
cd aula3_case_study/docker
docker-compose up -d bonsai
```

Then open: http://localhost:3000

### Run Standalone

```bash
cd aula3_case_study/api
python bonsai_app.py
```

Then open: http://localhost:3000

## Prompt Modes

BonsAI supports 4 different conversation modes:

1. **Basic**: Simple conversational bonsai advice
2. **Structured**: Problem-solution format with clear sections
3. **Diagnostic**: Systematic analysis for bonsai problems
4. **Emergency**: Urgent care for critical bonsai situations

## Environment Variables

Create a `.env` file with:

```env
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_KEY=your_api_key
OPENAI_API_VERSION=2024-12-01-preview
AZURE_DEPLOYMENT_NAME=gpt-4o
MLFLOW_TRACKING_URI=http://localhost:5000
```

## API Endpoints

- `GET /` - Chat interface
- `POST /chat` - Send messages to BonsAI
- `GET /health` - Health check
- `GET /bonsai/info` - Bot capabilities
- `GET /examples` - Example bonsai questions
- `POST /prompt/switch` - Change conversation mode

## Example Questions

Try asking BonsAI about:

- "How often should I water my Juniper bonsai?"
- "What soil mix is best for Ficus bonsai?"
- "My bonsai leaves are yellowing, help!"
- "When should I repot my bonsai?"
- "How to wire bonsai branches safely?"

## Notebook Integration

The chat bot is based on prompt engineering experiments from:
`notebooks/MLflow_3_3_1_Prompt_Engineering_Exploration.ipynb`

## Architecture

```
BonsAI Chat Bot
├── Web Interface (HTML/CSS/JS)
├── Flask API (Python)
├── Azure OpenAI (LLM)
├── MLflow (Tracking)
└── Specialized Bonsai Prompts
```

---

**Note**: BonsAI only answers questions about bonsai plants. For other plant care needs, use the general plant care API.
