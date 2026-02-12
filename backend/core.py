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

# Token limit constants (Llama 3.1 8B has 8K context)
MAX_MEMORY_ENTRIES = 3  # Keep only last 3 exchanges
MAX_MEMORY_SIZE_TOKENS = 1500  # Reserve for memory (rough estimate: ~4 chars per token)


def get_memory(session_id: str, limit: int = None):
    """Get conversation memory for a session.
    
    Args:
        session_id: Session identifier
        limit: Max number of entries (uses MAX_MEMORY_ENTRIES if None)
    
    Returns:
        List of memory entries limited to save tokens
    """
    if limit is None:
        limit = MAX_MEMORY_ENTRIES
    messages = memory_store.get(session_id, [])
    return messages[-limit:]


def format_memory_concise(memory: list) -> str:
    """Format memory in a concise way to save tokens.
    
    Args:
        memory: List of memory entries
    
    Returns:
        Formatted memory string suitable for LLM context
    """
    if not memory:
        return ""
    
    formatted = []
    for entry in memory:
        user_q = entry.get("user", "").strip()[:100]  # Truncate long queries
        ai_a = entry.get("ai", "").strip()[:150]  # Truncate long responses
        if user_q and ai_a:
            formatted.append(f"Q: {user_q}\nA: {ai_a}")
    
    return "\n---\n".join(formatted) if formatted else ""


def add_memory(session_id: str, user_text: str, ai_text: str, citations=None):
    """Add conversation turn to memory with token awareness.
    
    Args:
        session_id: Session identifier
        user_text: User query
        ai_text: AI response
        citations: List of citations for this exchange
    """
    memory_store.setdefault(session_id, []).append(
        {"user": user_text[:200], "ai": ai_text[:300], "citations": citations or []}
    )
