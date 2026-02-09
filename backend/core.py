import os
from dotenv import load_dotenv
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
chroma_client = chromadb.PersistentClient(path="chroma_db")
documents_collection = chroma_client.get_or_create_collection(name="documents")
groq_client = Groq(api_key=GROQ_API_KEY)

memory_store = {}

ALLOWED_FILTER_KEYS = {
    "type",
    "department",
    "employee_id",
    "category",
    "severity",
    "incident_id",
    "policy_id",
    "guide_id",
    "intent",
    "source_filename",
    "source_type",
}


def get_memory(session_id: str, limit: int = 6):
    messages = memory_store.get(session_id, [])
    if limit is None:
        return messages
    return messages[-limit:]


def add_memory(session_id: str, user_text: str, ai_text: str, citations=None):
    memory_store.setdefault(session_id, []).append(
        {"user": user_text, "ai": ai_text, "citations": citations or []}
    )
