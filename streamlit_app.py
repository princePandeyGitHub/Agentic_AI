import json
import uuid
from typing import Dict

import requests
import streamlit as st


def api_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"


def post_ingest(base: str, file_obj, intent: str, metadata_json: str):
    files = {"file": (file_obj.name, file_obj.getvalue())}
    data = {"intent": intent, "metadata_json": metadata_json}
    return requests.post(api_url(base, "/ingest/"), files=files, data=data, timeout=120)


def post_chat(base: str, payload: Dict):
    return requests.post(api_url(base, "/chat/"), json=payload, timeout=120)


st.set_page_config(page_title="Agentic RAG", layout="wide")
st.title("Agentic RAG System")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Backend")
    api_base = st.text_input("API base URL", "http://localhost:8000")
    session_id = st.text_input("Session ID", value=st.session_state.session_id)
    if st.button("Reset session"):
        st.session_state.session_id = str(uuid.uuid4())
        session_id = st.session_state.session_id
    st.caption("Keep the same Session ID to preserve chat memory.")

st.subheader("Upload & Ingest")
col1, col2 = st.columns([2, 1])

with col1:
    upload = st.file_uploader(
        "Upload a document (PDF, DOCX, TXT, JSON)",
        type=["pdf", "docx", "txt", "json"],
    )
with col2:
    intent = st.text_input("Intent label", value="general")
    metadata_json = st.text_area("Metadata JSON (optional)", value="")

if st.button("Ingest Document", type="primary", disabled=upload is None):
    try:
        response = post_ingest(api_base, upload, intent, metadata_json)
        if response.ok:
            st.success("Ingestion complete.")
            st.json(response.json())
        else:
            st.error(f"Ingestion failed ({response.status_code}).")
            st.text(response.text)
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")

st.divider()
st.subheader("Chat")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:
    role = message["role"]
    with st.chat_message(role):
        st.markdown(message["content"])
        if role == "assistant" and message.get("meta"):
            with st.expander("Details"):
                st.json(message["meta"])

prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {"query": prompt, "session_id": session_id, "n_results": 4}
    try:
        response = post_chat(api_base, payload)
        if response.ok:
            data = response.json()
            answer = data.get("answer", "")
            meta = {
                "intent": data.get("intent"),
                "grounding": data.get("grounding"),
                "filters": data.get("filters"),
                "citations": data.get("citations"),
            }
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": answer, "meta": meta}
            )
            with st.chat_message("assistant"):
                st.markdown(answer)
                with st.expander("Details"):
                    st.json(meta)
        else:
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": "Backend error."}
            )
            with st.chat_message("assistant"):
                st.error(f"Chat failed ({response.status_code}).")
                st.text(response.text)
    except requests.RequestException as exc:
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": "Request failed."}
        )
        with st.chat_message("assistant"):
            st.error(f"Request failed: {exc}")
