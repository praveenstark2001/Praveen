# Mental Health Chatbot

## Overview
The Mental Health Chatbot is an interactive application designed to provide users with information and support related to mental health issues. It leverages a fine-tuned version of the Llama 2 model for natural language understanding and response generation, ensuring that conversations are empathetic and informative.

## Features
- User authentication with streak tracking
- Conversational interface for mental health queries
- Fine-tuned Llama 2 model for accurate responses
- Vector store using FAISS for efficient retrieval of relevant information

- # Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
torch
transformers
datasets
streamlit
langchain
langchain-huggingface
langchain-community
faiss-cpu
Usage
Run the Chatbot: Start the Streamlit application.
bash
streamlit run mental_health_chatbot.py
Login: Use the provided username and password to access the chatbot.
Interact: Ask questions related to mental health and receive responses from the chatbot.
