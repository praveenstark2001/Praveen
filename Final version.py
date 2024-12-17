import os
import streamlit as st
import random
import time
import pandas as pd
import json
import torch
import warnings
from transformers import AutoTokenizer
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.ctransformers import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# File to store streaks
DATA_FILE = "user_streaks.json"

# Initialize streak file if it doesn't exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w") as f:
        json.dump({}, f)

# Load streak data
def load_streak_data():
    try:
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

# Save streak data
def save_streak_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Load and preprocess datasets
def load_and_combine_datasets():
    try:
        if not os.path.exists('combined_preprocessed_data.csv'):
            st.error("Primary dataset 'combined_preprocessed_data.csv' is missing.")
            return []

        primary_data = pd.read_csv('combined_preprocessed_data.csv')
        primary_data['text'] = primary_data['Context'].fillna('') + " " + primary_data['Response'].fillna('')
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_text(" ".join(primary_data['text'].tolist()))
        return texts
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return []

# Streamlit page configuration
st.set_page_config(
    page_title="Mental Health Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #2C2C34; /* Dark background */
        color: #FFFFFF; /* White text */
        font-family: 'Arial', sans-serif;
    }
    .header {
        background: linear-gradient(90deg, #3a6186, #89253e);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    }
    .quote-box {
        background-color: #424250; /* Soft gray */
        font-style: italic;
        font-size: 18px;
        color: #E0E0E0;
        padding: 15px;
        border-left: 5px solid #6A11CB;
        margin: 20px 0;
        border-radius: 8px;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
    }
    .chat-bubble-user {
        background: linear-gradient(90deg, #3a6186, #89253e);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 60%;
        align-self: flex-end;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
    }
    .chat-bubble-bot {
        background: linear-gradient(90deg, #6A11CB, #2575FC);
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        max-width: 60%;
        align-self: flex-start;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "conversation" not in st.session_state:
    st.session_state["conversation"] = []
if "history" not in st.session_state:
    st.session_state["history"] = []
if "generated" not in st.session_state:
    st.session_state["generated"] = ["Hello! 😊 Welcome to the Mental Health Chatbot. Ask me anything about mental health."]
if "past" not in st.session_state:
    st.session_state["past"] = ["Hey there! 👋"]

# Login function
def login():
    st.title("Login to Mental Health Chatbot")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    age = st.text_input("Age", placeholder="Enter your age")
    gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other"])

    if st.button("Login"):
        if username and password and gender != "Select":
            streak_data = load_streak_data()

            # Increment streak for the user or initialize it
            if username not in streak_data:
                streak_data[username] = {"streak": 1}
            else:
                streak_data[username]["streak"] += 1

            save_streak_data(streak_data)

            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["age"] = age
            st.session_state["gender"] = gender
            st.success(f"Welcome back, {username}!")
        else:
            st.error("Invalid credentials or missing information. Please try again.")

# Load embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
try:
    if os.path.exists("faiss_index"):
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        new_texts = load_and_combine_datasets()
        if new_texts:
            vector_store.add_texts(new_texts)
    else:
        texts = load_and_combine_datasets()
        if texts:
            vector_store = FAISS.from_texts(texts, embeddings)
            vector_store.save_local("faiss_index")
except Exception as e:
    st.error(f"Error handling FAISS index: {str(e)}")

# Create LLM model
llm = CTransformers(
    model="/path/to/your/fine-tuned/llama-2-model.bin",
    model_type="llama",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    config={"max_new_tokens": 1024, "temperature": 0.7}
)

# Conversational retrieval chain setup
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type='stuff',
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

# Function to handle user input and display conversation history
def conversation_chat(query):
    try:
        full_query = f"[User: {st.session_state['gender']}, Age: {st.session_state['age']}] {query}"
        result = chain.invoke({"question": full_query, "chat_history": st.session_state['history']})
        response = result["answer"].replace("Please answer the question:", "").strip()
        st.session_state['history'].append((query, response))
        st.session_state['past'].append(query)
        st.session_state['generated'].append(response)

        return response
    except Exception as e:
        st.error(f"Error in chatbot response: {str(e)}")
        return "I'm sorry, I couldn't process that. Please try again."

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Ask a question about mental health:", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                if i < len(st.session_state["past"]):
                    st.markdown(f'<div class="chat-bubble-user">{st.session_state["past"][i]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="chat-bubble-bot">{st.session_state["generated"][i]}</div>', unsafe_allow_html=True)

# Main
if not st.session_state["logged_in"]:
    login()
else:
    display_chat_history()