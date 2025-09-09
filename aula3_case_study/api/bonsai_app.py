"""
BonsAI Chat Bot - A specialized bonsai care assistant
Interactive web interface for bonsai plant care assistance using MLflow and Azure OpenAI
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
import mlflow
import requests
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, List
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
EXPERIMENT_NAME = 'Bonsai-Care-Prompt-Engineering'

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_API_VERSION = os.getenv('OPENAI_API_VERSION', '2024-12-01-preview')
DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME', 'gpt-4o')

# Initialize Azure OpenAI client
azure_client = None
if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY:
    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
    )

# BonsAI specific prompt templates
BONSAI_PROMPTS = {
    "basic": {
        "name": "plant_care_basic",
        "template": """You are BonsAI, a specialized bonsai care expert assistant. You only provide information related to bonsai plants and their care. If someone asks about anything other than bonsai, politely redirect them back to bonsai topics.

Customer Question: {query}

Answer as BonsAI:""",
        "description": "Basic bonsai care assistant prompt"
    },
    
    "structured": {
        "name": "bonsai_care_structured", 
        "template": """You are BonsAI, a professional bonsai care consultant. You only answer questions about bonsai plants. Provide a structured response to the customer's bonsai care question.

Customer Question: {query}

Please structure your response as follows:
1. **Problem Assessment**: Brief analysis of the bonsai issue
2. **Immediate Actions**: What to do right now for your bonsai
3. **Long-term Care**: Ongoing bonsai care recommendations
4. **Prevention**: How to prevent this bonsai issue in the future

If the question is not about bonsai, politely say: "I'm sorry, but I can only provide information related to bonsai plants."

BonsAI Response:""",
        "description": "Structured bonsai care response format"
    },
    
    "diagnostic": {
        "name": "bonsai_care_diagnostic",
        "template": """You are BonsAI, a bonsai pathologist assistant. Help diagnose bonsai problems systematically. You only provide assistance with bonsai plants.

Customer Description: {query}

Analysis Process for Bonsai:
1. Identify key symptoms mentioned in the bonsai
2. Consider possible causes (watering, light, nutrients, pests, diseases specific to bonsai)
3. Ask clarifying questions about the bonsai if needed
4. Provide bonsai diagnosis with confidence level
5. Suggest bonsai treatment plan

If the question is not about bonsai, respond: "I'm sorry, but I can only provide information related to bonsai plants."

BonsAI Diagnostic Response:""",
        "description": "Diagnostic approach for bonsai problems"
    },
    
    "emergency": {
        "name": "bonsai_care_emergency",
        "template": """🌿 BONSAI EMERGENCY RESPONSE PROTOCOL 🌿

You are BonsAI, an emergency bonsai care specialist. The customer has an urgent bonsai problem that needs immediate attention. You only help with bonsai emergencies.

Emergency Description: {query}

IMMEDIATE BONSAI RESPONSE PROTOCOL:
⚡ URGENT ACTIONS (Next 24 hours for your bonsai):
🔍 BONSAI ASSESSMENT NEEDED:
📋 BONSAI MONITORING PLAN:
⚠️  WARNING SIGNS TO WATCH IN YOUR BONSAI:

If this is not a bonsai emergency, respond: "I'm sorry, but I can only provide information related to bonsai plants."

Provide quick, actionable advice to save your bonsai!""",
        "description": "Emergency response for critical bonsai issues"
    }
}

# Global variables
current_prompt_template = BONSAI_PROMPTS["basic"]
model_info = {}

# HTML Template for the chat interface
CHAT_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BonsAI Chat - Your Bonsai Care Expert</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 90vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        
        .chat-header h1 {
            font-size: 1.8em;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 0.9em;
        }
        
        .status-indicator {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 12px;
            height: 12px;
            background: #4CAF50;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.bot {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            font-size: 0.9em;
            line-height: 1.4;
        }
        
        .message.user .message-content {
            background: #007bff;
            color: white;
            border-bottom-right-radius: 4px;
        }
        
        .message.bot .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 4px;
        }
        
        .message-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
            font-size: 0.8em;
        }
        
        .message.user .message-avatar {
            background: #007bff;
            order: 2;
        }
        
        .message.bot .message-avatar {
            background: #4CAF50;
            order: 1;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        .input-group input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 0.9em;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .input-group input:focus {
            border-color: #4CAF50;
        }
        
        .input-group button {
            padding: 12px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s;
        }
        
        .input-group button:hover {
            background: #45a049;
        }
        
        .input-group button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .welcome-message {
            text-align: center;
            color: #666;
            padding: 40px 20px;
        }
        
        .welcome-message h3 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        
        .quick-questions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 15px;
            justify-content: center;
        }
        
        .quick-question {
            background: #e8f5e8;
            color: #2e7d2e;
            padding: 8px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            cursor: pointer;
            transition: background 0.3s;
            border: none;
        }
        
        .quick-question:hover {
            background: #4CAF50;
            color: white;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px 16px;
            background: white;
            border-radius: 18px;
            border: 1px solid #e0e0e0;
            width: fit-content;
            margin-bottom: 15px;
        }
        
        .typing-dots {
            display: inline-block;
        }
        
        .typing-dots span {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #4CAF50;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
        .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        .error-message {
            color: #dc3545;
            text-align: center;
            padding: 10px;
            background: #f8d7da;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="status-indicator"></div>
            <h1>🌿 BonsAI Chat</h1>
            <p>Your specialized bonsai care expert assistant</p>
        </div>
        
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <h3>Welcome to BonsAI!</h3>
                <p>I'm your specialized bonsai care expert. Ask me anything about bonsai care, styling, watering, fertilizing, or any bonsai-related questions!</p>
                <div class="quick-questions">
                    <button class="quick-question" onclick="sendQuickQuestion(this)">How often should I water my Juniper bonsai?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">What soil mix is best for Ficus bonsai?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">My bonsai leaves are yellowing, help!</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">When should I repot my bonsai?</button>
                    <button class="quick-question" onclick="sendQuickQuestion(this)">How to wire bonsai branches?</button>
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
            BonsAI is thinking...
        </div>
        
        <div class="chat-input">
            <div class="input-group">
                <input type="text" id="messageInput" placeholder="Ask me about your bonsai..." onkeypress="handleKeyPress(event)">
                <button onclick="sendMessage()" id="sendButton">Send</button>
            </div>
        </div>
    </div>

    <script>
        const chatMessages = document.getElementById('chatMessages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        
        function scrollToBottom() {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        function addMessage(content, isUser = false) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = isUser ? 'You' : '🌿';
            
            const messageContent = document.createElement('div');
            messageContent.className = 'message-content';
            messageContent.innerHTML = content.replace(/\\n/g, '<br>');
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(messageContent);
            
            chatMessages.appendChild(messageDiv);
            scrollToBottom();
        }
        
        function showTyping() {
            typingIndicator.style.display = 'block';
            scrollToBottom();
        }
        
        function hideTyping() {
            typingIndicator.style.display = 'none';
        }
        
        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = message;
            chatMessages.appendChild(errorDiv);
            scrollToBottom();
        }
        
        async function sendMessage() {
            const message = messageInput.value.trim();
            if (!message) return;
            
            // Clear welcome message if it exists
            const welcomeMessage = document.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }
            
            // Add user message
            addMessage(message, true);
            messageInput.value = '';
            sendButton.disabled = true;
            showTyping();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: message })
                });
                
                if (!response.ok) {
                    throw new Error(`Server error: ${response.status}`);
                }
                
                const data = await response.json();
                hideTyping();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    addMessage(data.response);
                }
                
            } catch (error) {
                hideTyping();
                showError('Sorry, I encountered an error. Please try again.');
                console.error('Error:', error);
            } finally {
                sendButton.disabled = false;
                messageInput.focus();
            }
        }
        
        function sendQuickQuestion(button) {
            messageInput.value = button.textContent;
            sendMessage();
        }
        
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        // Focus on input when page loads
        messageInput.focus();
    </script>
</body>
</html>
"""

def load_bonsai_prompt(prompt_type="basic"):
    """Load the appropriate bonsai prompt template - first try MLflow, then fallback to local"""
    global current_prompt_template, model_info
    
    # First, try to load from MLflow prompt registry
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"🔗 Connecting to MLflow at {MLFLOW_TRACKING_URI}")
        
        # Map prompt types to MLflow registry names
        mlflow_prompt_names = {
            "basic": "plant_care_basic",
            "structured": "bonsai_care_structured", 
            "diagnostic": "bonsai_care_diagnostic",
            "emergency": "bonsai_care_emergency"
        }
        
        prompt_name = mlflow_prompt_names.get(prompt_type, "plant_care_basic")
        
        # Try to load the prompt from MLflow registry
        try:
            prompt_uri = f"prompts:/{prompt_name}/latest"
            registered_prompt = mlflow.genai.load_prompt(prompt_uri)
            
            # Create template structure compatible with our app
            current_prompt_template = {
                "name": prompt_name,
                "template": registered_prompt.template,
                "description": f"MLflow registered prompt: {prompt_name}"
            }
            
            model_info = {
                "prompt_name": prompt_name,
                "prompt_type": prompt_type,
                "loaded_at": datetime.now().isoformat(),
                "status": "mlflow_registered",
                "description": f"Loaded from MLflow registry: {prompt_uri}",
                "source": "mlflow"
            }
            
            logger.info(f"✅ Loaded BonsAI prompt from MLflow: {prompt_name}")
            return True
            
        except Exception as mlflow_error:
            logger.warning(f"⚠️ Could not load {prompt_name} from MLflow: {mlflow_error}")
            # Fall back to local prompts
            
    except Exception as e:
        logger.warning(f"⚠️ MLflow connection failed: {e}")
    
    # Fallback to local prompt templates
    if prompt_type in BONSAI_PROMPTS:
        current_prompt_template = BONSAI_PROMPTS[prompt_type]
        model_info = {
            "prompt_name": current_prompt_template["name"],
            "prompt_type": prompt_type,
            "loaded_at": datetime.now().isoformat(),
            "status": "local_fallback",
            "description": current_prompt_template["description"],
            "source": "local"
        }
        logger.info(f"✅ Loaded BonsAI prompt from local templates: {prompt_type}")
        return True
    
    logger.warning(f"⚠️ Unknown prompt type: {prompt_type}, using basic")
    return load_bonsai_prompt("basic")

def query_azure_openai(prompt: str) -> str:
    """Send query to Azure OpenAI with BonsAI personality"""
    try:
        if not azure_client:
            logger.error("Azure OpenAI client not initialized")
            return "Sorry, I'm not properly configured right now. Please check my Azure OpenAI settings."
            
        response = azure_client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        
        if response.choices:
            return response.choices[0].message.content
        else:
            logger.error("No response from Azure OpenAI")
            return "Sorry, I couldn't generate a response right now."
            
    except Exception as e:
        logger.error(f"Error querying Azure OpenAI: {str(e)}")
        return "Sorry, I encountered an error while processing your bonsai question. Please try again."

@app.route('/', methods=['GET'])
def chat_interface():
    """Serve the main chat interface"""
    return render_template_string(CHAT_HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "bonsai-chat-bot",
        "bot_name": "BonsAI",
        "model_info": model_info,
        "azure_openai_status": "connected" if azure_client else "not_configured"
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint for BonsAI assistance"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' field in request"
            }), 400
        
        user_query = data['query']
        
        # Validate input
        if not user_query.strip():
            return jsonify({
                "error": "Query cannot be empty"
            }), 400
        
        # Apply BonsAI prompt template
        formatted_prompt = current_prompt_template['template'].format(query=user_query)
        
        # Query the LLM
        logger.info(f"🌿 BonsAI processing query: {user_query[:50]}...")
        ai_response = query_azure_openai(formatted_prompt)
        
        # Prepare response
        response_data = {
            "query": user_query,
            "response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "bot_name": "BonsAI",
            "model": DEPLOYMENT_NAME,
            "prompt_template": current_prompt_template['name']
        }
        
        logger.info(f"✅ BonsAI response generated successfully")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Error in BonsAI chat endpoint: {str(e)}")
        return jsonify({
            "error": "I'm having trouble right now. Please try again.",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/prompt/switch', methods=['POST'])
def switch_prompt():
    """Switch to a different prompt template"""
    try:
        data = request.get_json()
        prompt_type = data.get('prompt_type', 'basic')
        
        if load_bonsai_prompt(prompt_type):
            return jsonify({
                "status": "success",
                "message": f"Switched to {prompt_type} prompt",
                "current_prompt": current_prompt_template,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": "Failed to switch prompt template"
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Error switching prompt: {str(e)}")
        return jsonify({
            "error": "Internal server error"
        }), 500

@app.route('/prompt/reload', methods=['POST'])
def reload_prompt_from_mlflow():
    """Reload the current prompt from MLflow registry"""
    try:
        current_type = model_info.get('prompt_type', 'basic')
        
        if load_bonsai_prompt(current_type):
            return jsonify({
                "status": "success",
                "message": f"Reloaded {current_type} prompt from MLflow",
                "current_prompt": current_prompt_template,
                "model_info": model_info,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "error": "Failed to reload prompt from MLflow"
            }), 400
            
    except Exception as e:
        logger.error(f"❌ Error reloading prompt: {str(e)}")
        return jsonify({
            "error": "Internal server error"
        }), 500

@app.route('/prompt/info', methods=['GET'])
def prompt_info():
    """Get information about the current prompt template"""
    return jsonify({
        "current_prompt": current_prompt_template,
        "available_prompts": list(BONSAI_PROMPTS.keys()),
        "model_info": model_info,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/prompt/register', methods=['POST'])
def register_prompts_to_mlflow():
    """Register local prompt templates to MLflow registry"""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        registered_prompts = []
        failed_prompts = []
        
        for prompt_type, prompt_config in BONSAI_PROMPTS.items():
            try:
                client = mlflow.tracking.MlflowClient()
                
                # Register the prompt
                registered_prompt = client.register_prompt(
                    name=prompt_config["name"],
                    template=prompt_config["template"],
                    tags={"type": prompt_type, "domain": "bonsai_care", "source": "bonsai_app"}
                )
                
                registered_prompts.append({
                    "name": prompt_config["name"],
                    "version": registered_prompt.version,
                    "type": prompt_type
                })
                
                logger.info(f"✅ Registered prompt: {prompt_config['name']} (Version {registered_prompt.version})")
                
            except Exception as e:
                failed_prompts.append({
                    "name": prompt_config["name"],
                    "type": prompt_type,
                    "error": str(e)
                })
                logger.error(f"❌ Failed to register {prompt_config['name']}: {e}")
        
        return jsonify({
            "status": "completed",
            "registered_prompts": registered_prompts,
            "failed_prompts": failed_prompts,
            "total_registered": len(registered_prompts),
            "total_failed": len(failed_prompts),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error in prompt registration: {str(e)}")
        return jsonify({
            "error": "Failed to register prompts to MLflow",
            "details": str(e)
        }), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_response():
    """Evaluate a BonsAI response for feedback"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data or 'response' not in data or 'rating' not in data:
            return jsonify({
                "error": "Missing required fields: 'query', 'response', 'rating'"
            }), 400
        
        # Log evaluation data
        evaluation_data = {
            "query": data['query'],
            "response": data['response'], 
            "rating": data['rating'],
            "feedback": data.get('feedback', ''),
            "timestamp": datetime.now().isoformat(),
            "bot_name": "BonsAI",
            "prompt_template": current_prompt_template['name']
        }
        
        logger.info(f"📊 BonsAI evaluation received: Rating {data['rating']}/5")
        
        return jsonify({
            "status": "evaluation_recorded",
            "message": "Thank you for your feedback! This helps BonsAI learn.",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"❌ Error in evaluate endpoint: {str(e)}")
        return jsonify({
            "error": "Internal server error"
        }), 500

@app.route('/bonsai/info', methods=['GET'])
def bonsai_info():
    """Get BonsAI bot information and capabilities"""
    return jsonify({
        "bot_name": "BonsAI",
        "description": "Specialized bonsai care expert assistant",
        "capabilities": [
            "Bonsai care advice",
            "Species-specific guidance",
            "Watering schedules",
            "Soil recommendations",
            "Styling techniques",
            "Problem diagnosis",
            "Emergency care"
        ],
        "specialization": "Bonsai plants only",
        "available_prompt_modes": {
            "basic": "Simple conversational bonsai advice",
            "structured": "Structured problem-solution format",
            "diagnostic": "Systematic bonsai problem analysis",
            "emergency": "Urgent bonsai care situations"
        },
        "current_mode": current_prompt_template['name'],
        "model": DEPLOYMENT_NAME,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/examples', methods=['GET'])
def example_queries():
    """Get example bonsai questions for testing"""
    examples = [
        {
            "query": "How often should I water my Juniper bonsai?",
            "category": "care_routine",
            "difficulty": "beginner"
        },
        {
            "query": "What is the best soil mix for a Ficus bonsai?",
            "category": "soil_care",
            "difficulty": "beginner"
        },
        {
            "query": "My bonsai's leaves are turning yellow and falling off. What should I do?",
            "category": "problem_diagnosis",
            "difficulty": "intermediate"
        },
        {
            "query": "What does the word 'bonsai' mean?",
            "category": "general_knowledge",
            "difficulty": "beginner"
        },
        {
            "query": "Can I keep my bonsai tree indoors?",
            "category": "care_environment",
            "difficulty": "beginner"
        },
        {
            "query": "What is nebari in bonsai?",
            "category": "techniques",
            "difficulty": "intermediate"
        },
        {
            "query": "How do I wire bonsai branches safely?",
            "category": "styling",
            "difficulty": "advanced"
        },
        {
            "query": "When should I repot my pine bonsai?",
            "category": "care_routine",
            "difficulty": "intermediate"
        }
    ]
    
    return jsonify({
        "example_queries": examples,
        "total_examples": len(examples),
        "instructions": "Send POST request to /chat with {'query': 'your bonsai question'}",
        "note": "BonsAI only answers questions about bonsai plants"
    })

# Initialize the application
def initialize_app():
    """Initialize the BonsAI Flask application"""
    logger.info("🌿 Initializing BonsAI Chat Bot")
    
    # Load the default bonsai prompt template (will try MLflow first)
    if not load_bonsai_prompt("basic"):
        logger.warning("⚠️ Using fallback configuration")
    
    # Try to register local prompts to MLflow if they don't exist
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Check if basic prompt exists, if not register all prompts
        try:
            client.get_registered_prompt("plant_care_basic")
            logger.info("📝 BonsAI prompts already registered in MLflow")
        except:
            logger.info("📝 Failed to get BonsAI prompts to MLflow...")
            # for prompt_type, prompt_config in BONSAI_PROMPTS.items():
            #     try:
            #         registered_prompt = client.register_prompt(
            #             name=prompt_config["name"],
            #             template=prompt_config["template"],
            #             tags={"type": prompt_type, "domain": "bonsai_care", "source": "bonsai_app"}
            #         )
            #         logger.info(f"✅ Registered: {prompt_config['name']} (v{registered_prompt.version})")
            #     except Exception as e:
            #         logger.warning(f"⚠️ Could not register {prompt_config['name']}: {e}")
                    
    except Exception as e:
        logger.warning(f"⚠️ MLflow prompt registration skipped: {e}")
    
    # Validate Azure OpenAI configuration
    if not azure_client:
        logger.warning("⚠️ Azure OpenAI not properly configured")
    else:
        logger.info("✅ Azure OpenAI client initialized")
    
    logger.info("✅ BonsAI Chat Bot initialized successfully")
    logger.info("🌐 Access the chat interface at: http://localhost:3000")

if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=3000, debug=True)
