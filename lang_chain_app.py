"""
Streamlit frontend for "Ramya's Bot" — AI Chat Helper using Google Gemini API (AI Studio)

Features:
- Conversation UI with history
- API key entry (for Google AI Studio)
- System prompt + user messages
- Optional file upload to add context

Usage:
1. Install requirements:
   pip install streamlit google-generativeai python-dotenv

2. Run:
   streamlit run Ramya_Bot_Streamlit_Frontend.py

3. Provide your Google AI Studio (Gemini) API key in the sidebar or set environment variable GOOGLE_API_KEY.
"""

import streamlit as st
import google.generativeai as genai
import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

# ---------- Configuration ----------
DEFAULT_MODEL = "gemini-1.5-flash"
MAX_TOKENS = 1024

# ---------- Helper functions ----------

def set_api_key(key: str):
    os.environ["GOOGLE_API_KEY"] = key
    genai.configure(api_key=key)


def get_api_key_from_env():
    return os.environ.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")


def format_messages_for_prompt(chat_history: List[Dict[str, str]], system_prompt: str = None):
    prompt = ""
    if system_prompt:
        prompt += f"System: {system_prompt}\n"
    for item in chat_history:
        role = item["role"].capitalize()
        prompt += f"{role}: {item['content']}\n"
    prompt += "Assistant:"
    return prompt


def call_gemini_chat(messages: List[Dict[str, str]], model=DEFAULT_MODEL, temperature=0.2):
    key = get_api_key_from_env()
    if not key:
        raise ValueError("No Google AI API key provided. Set GOOGLE_API_KEY or enter it in the sidebar.")

    genai.configure(api_key=key)
    chat = genai.GenerativeModel(model)

    prompt = format_messages_for_prompt(messages)

    response = chat.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
        )
    )
    return response.text.strip()


# ---------- Streamlit UI ----------

st.set_page_config(page_title="Ramya's Bot — AI Chat Helper", layout="wide")
st.title("🤖 Ramya's Bot — AI Chat Helper (Gemini API)")

# Sidebar: settings
with st.sidebar:
    st.header("Settings & API")
    api_key_input = st.text_input("Google AI Studio API Key (Gemini)", type="password")
    if api_key_input:
        set_api_key(api_key_input)
        st.success("API key set for this session")

    model = st.selectbox("Model", options=["gemini-2.5-flash", "gemini-1.5-pro"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    st.markdown("---")
    st.markdown("**Optional:** Upload a text file to give the bot extra context (it will be appended to system prompt).")
    uploaded_file = st.file_uploader("Upload .txt or .md file", type=["txt", "md"], accept_multiple_files=False)

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant. Answer user questions concisely but completely." 

# If uploaded file present, read and add to system prompt
if uploaded_file is not None:
    try:
        file_text = uploaded_file.getvalue().decode("utf-8")
    except Exception:
        file_text = ""
    if file_text:
        st.session_state.system_prompt += "\n\n" + file_text
        st.info("Uploaded file appended to system prompt")

# Display conversation
st.subheader("Conversation")
chat_container = st.container()

with chat_container:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**Ramya's Bot:** {msg['content']}")
        elif msg["role"] == "system":
            st.markdown(f"*System:* {msg['content']}")

# Input
st.markdown("---")
user_input = st.text_input("Type your message and press Enter", key="user_input_field")

if st.button("Clear Conversation"):
    st.session_state.history = []
    st.success("Conversation cleared")

if user_input:
    # Append user message
    st.session_state.history.append({"role": "user", "content": user_input})

    # Prepare history for model
    history_for_model = []
    history_for_model.append({"role": "system", "content": st.session_state.system_prompt})
    for m in st.session_state.history[:-1]:
        history_for_model.append({"role": m["role"], "content": m["content"]})

    with st.spinner("Ramya's Bot is thinking..."):
        try:
            messages = history_for_model + [{"role": "user", "content": user_input}]
            answer = call_gemini_chat(messages, model=model, temperature=temperature)
        except Exception as e:
            st.error(f"Error from backend: {e}")
            answer = "I'm sorry — I couldn't get a response. Check your API key and network."

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()

# Footer / tips
st.markdown("---")
st.markdown("**Tips:** Ramya's Bot uses Google Gemini (AI Studio). You can upload custom text files to provide context for better answers.")
