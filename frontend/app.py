import base64

import httpx
import streamlit as st

st.set_page_config(page_title="Talking to Bridges", layout="wide")

API_BASE_URL = "http://127.0.0.1:8000"

st.title("Talking to Bridges")
st.subheader("An LLM-Based Intelligent Interface for Structural Health Monitoring")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for document upload
with st.sidebar:
    st.header("Groq Connection")
    try:
        health_response = httpx.get(f"{API_BASE_URL}/health", timeout=5.0)
        if health_response.status_code == 200 and health_response.json().get(
            "groq_connected", False
        ):
            st.success("Groq connected")
        else:
            st.warning("Groq is not connected. Add GROQ_API_KEY to .env.")
    except httpx.HTTPError:
        st.warning("Backend is not running. Start the API before using Groq.")

    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])
    if st.button("Upload Document") and uploaded_file is not None:
        with st.spinner("Uploading..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = httpx.post(f"{API_BASE_URL}/upload", files=files, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    st.session_state.messages = []
                    st.success(f"Uploaded! Document ID: {data.get('document_id')}")
                    st.info(f"Chunks created: {data.get('chunks_created')}")
                else:
                    st.error(f"Upload failed: {response.text}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Error connecting to server: {e}")

# Main chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "audio" in msg:
            st.audio(msg["audio"], format="audio/wav")

if prompt := st.chat_input("Ask a question about the bridge data..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call backend for chat response
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            chat_resp = httpx.post(
                f"{API_BASE_URL}/api/chat",
                json={"message": prompt},
                timeout=60.0
            )
            if chat_resp.status_code == 200:
                answer = chat_resp.json().get("answer", "No answer provided.")
                st.markdown(answer)
                
                # Call /speak for audio
                speak_resp = httpx.post(
                    f"{API_BASE_URL}/speak",
                    json={"message": answer},
                    timeout=30.0
                )
                audio_bytes = None
                if speak_resp.status_code == 200:
                    audio_base64 = speak_resp.json().get("audio_base64")
                    if audio_base64:
                        audio_bytes = base64.b64decode(audio_base64)
                        st.audio(audio_bytes, format="audio/wav")
                
                # Store to history
                msg_data = {"role": "assistant", "content": answer}
                if audio_bytes:
                    msg_data["audio"] = audio_bytes
                st.session_state.messages.append(msg_data)
            else:
                # Handle API error according to fixed format
                error_data = chat_resp.json()
                err_msg = error_data.get("error", {}).get("message", chat_resp.text)
                st.error(f"API Error: {err_msg}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to connect to backend API: {e}")
