#!/usr/bin/env python3
"""
Test script for BonsAI Chat Bot
"""

import requests
import json
import time
import sys

# Configuration
BONSAI_URL = "http://localhost:3000"  # BonsAI now runs on port 3000 (safe port)
TEST_QUESTIONS = [
    "How often should I water my Juniper bonsai?",
    "What soil mix is best for Ficus bonsai?", 
    "My bonsai leaves are yellowing, help!",
    "What does the word 'bonsai' mean?",
    "Can you help me with my regular houseplant?",  # Should refuse this
    "How do I wire bonsai branches safely?"
]

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f"{BONSAI_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data.get('bot_name', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_chat(question):
    """Test chat endpoint with a question"""
    try:
        response = requests.post(
            f"{BONSAI_URL}/chat",
            json={"query": question},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n🌿 Question: {question}")
            print(f"🤖 BonsAI: {data.get('response', 'No response')}")
            return True
        else:
            print(f"❌ Chat failed for '{question}': {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Chat error for '{question}': {e}")
        return False

def test_mlflow_integration():
    """Test MLflow prompt integration"""
    try:
        # Test prompt registration
        response = requests.post(f"{BONSAI_URL}/prompt/register")
        if response.status_code == 200:
            data = response.json()
            print(f"\n📝 Prompt Registration:")
            print(f"   Registered: {data.get('total_registered', 0)} prompts")
            print(f"   Failed: {data.get('total_failed', 0)} prompts")
            
        # Test prompt info
        response = requests.get(f"{BONSAI_URL}/prompt/info")
        if response.status_code == 200:
            data = response.json()
            print(f"\n📋 Current Prompt Info:")
            print(f"   Name: {data.get('model_info', {}).get('prompt_name', 'Unknown')}")
            print(f"   Source: {data.get('model_info', {}).get('source', 'Unknown')}")
            print(f"   Status: {data.get('model_info', {}).get('status', 'Unknown')}")
            return True
        else:
            print(f"❌ Prompt info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ MLflow integration test error: {e}")
        return False

def test_bonsai_info():
    """Test bonsai info endpoint"""
    try:
        response = requests.get(f"{BONSAI_URL}/bonsai/info")
        if response.status_code == 200:
            data = response.json()
            print(f"\n📋 BonsAI Info:")
            print(f"   Bot Name: {data.get('bot_name')}")
            print(f"   Description: {data.get('description')}")
            print(f"   Current Mode: {data.get('current_mode')}")
            return True
        else:
            print(f"❌ BonsAI info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ BonsAI info error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing BonsAI Chat Bot")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ BonsAI is not running or not healthy")
        print("🔧 Please start BonsAI with: python bonsai_app.py")
        sys.exit(1)
    
    # Test bonsai info
    test_bonsai_info()
    
    # Test MLflow integration
    test_mlflow_integration()
    
    # Test chat questions
    print(f"\n💬 Testing {len(TEST_QUESTIONS)} questions...")
    successful_tests = 0
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n[{i}/{len(TEST_QUESTIONS)}]", end="")
        if test_chat(question):
            successful_tests += 1
        time.sleep(1)  # Small delay between requests
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {successful_tests}/{len(TEST_QUESTIONS)} passed")
    
    if successful_tests == len(TEST_QUESTIONS):
        print("🎉 All tests passed! BonsAI is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above.")
    
    print(f"\n🌐 Access BonsAI web interface at: {BONSAI_URL}")

if __name__ == "__main__":
    main()
