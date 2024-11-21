import streamlit as st
import requests

# Configuration
API_URL = "http://localhost:8000/query/"  # Replace with your FastAPI /query/ endpoint

# Streamlit App Layout
st.set_page_config(page_title="RAG Chat App", page_icon="💬", layout="wide")
st.title("💬 Retrieval-Augmented Generation (RAG) Chat")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Function to send prompt to FastAPI and get response
def get_response(prompt):
    # headers = {"Content-Type": "application/json", "X-API-KEY": API_KEY}
    payload = {"prompt": prompt}
    try:
        response = requests.post(
            API_URL,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response received.")
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err} - {response.text}"
    except Exception as err:
        return f"An error occurred: {err}"


# Chat Interface
def chat_interface():
    st.markdown("### Chat with the RAG System")

    # Display chat history
    for message in st.session_state["messages"]:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**RAG System:** {message['content']}")

    # Input area for user prompt
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_area("Enter your message:", height=100)
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input.strip():
        # Append user message to chat history
        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Display user message immediately
        st.markdown(f"**You:** {user_input}")

        # Get response from API
        with st.spinner("Processing..."):
            response = get_response(user_input)

        # Append response to chat history
        st.session_state["messages"].append({"role": "assistant", "content": response})

        # Display response
        st.markdown(f"**RAG System:** {response}")


# Button to clear chat history
def clear_chat():
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()


# Render chat interface and clear button
chat_interface()
st.sidebar.markdown("---")
clear_chat()
