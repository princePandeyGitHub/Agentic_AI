import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.core import (
    ALLOWED_FILTER_KEYS,
    add_memory,
    documents_collection,
    get_memory,
    groq_client,
    model,
    format_memory_concise,
)

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    n_results: int = 4


class Citation(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    intent: str
    user_query: str
    filters: Dict[str, Any]
    grounding: str
    answer: str
    citations: List[Citation]


def extract_intent_and_filters(query: str) -> Dict[str, Any]:
    prompt = (
        "You are a classifier. Return ONLY JSON with keys "
        '"user_query", "intent", "filters". '
        "The intent should be a short label like hr, incident, general. "
        "Filters must be a JSON object. Use ONLY these keys if relevant: "
        f"{sorted(ALLOWED_FILTER_KEYS)}. "
        "If unsure, return an empty filters object. Do not include markdown."
        f"\nUser query: {query}"
    )
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=256,
    )
    raw = response.choices[0].message.content.strip()
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload_not_dict")
    except Exception:
        payload = {"user_query": query, "intent": "general", "filters": {}}
    payload.setdefault("user_query", query)
    payload.setdefault("intent", "general")
    if not isinstance(payload.get("filters"), dict):
        payload["filters"] = {}
    payload.setdefault("filters", {})
    return payload


def build_where(intent: str, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = []
    if intent and intent.lower() != "general":
        clauses.append({"intent": intent})
    for key, value in filters.items():
        if key in ALLOWED_FILTER_KEYS:
            clauses.append({key: value})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def build_intent_only_where(intent: str) -> Optional[Dict[str, Any]]:
    if intent and intent.lower() != "general":
        return {"intent": intent}
    return None


def answer_with_context(user_query: str, context: str, memory: List[Dict[str, str]]) -> str:
    memory_str = format_memory_concise(memory)
    memory_section = f"\nRecent conversation:\n{memory_str}\n" if memory_str else ""
    
    prompt = f"""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't know."{memory_section}
Context:
{context}

User question:
{user_query}

Answer concisely in 2-3 sentences:
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    return response.choices[0].message.content


def answer_with_memory(user_query: str, memory: List[Dict[str, str]]) -> str:
    memory_str = format_memory_concise(memory)
    
    prompt = f"""You are a helpful assistant answering follow-up questions about previously discussed documents.
Use ONLY the conversation history below. If the answer is not in the history, say "I don't know."

Conversation history:
{memory_str}

User question:
{user_query}

Answer concisely in 1-2 sentences:
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256,
    )
    return response.choices[0].message.content


def latest_memory_citations(memory: List[Dict[str, Any]]) -> List[Citation]:
    for entry in reversed(memory):
        citations = entry.get("citations") or []
        if citations:
            return [Citation(**item) for item in citations]
    return []


@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    extracted = extract_intent_and_filters(request.query)
    user_query = extracted.get("user_query", request.query)
    intent = str(extracted.get("intent", "general")).strip().lower()
    filters = extracted.get("filters", {})

    where = build_where(intent, filters)
    query_embedding = model.encode(user_query)
    results = documents_collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=request.n_results,
        where=where,
    )

    documents = results.get("documents", [[]])[0] or []
    ids = results.get("ids", [[]])[0] or []
    metadatas = results.get("metadatas", [[]])[0] or []
    citations = []

    if not documents:
        intent_where = build_intent_only_where(intent)
        if intent_where != where:
            results = documents_collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=request.n_results,
                where=intent_where,
            )
            documents = results.get("documents", [[]])[0] or []
            ids = results.get("ids", [[]])[0] or []
            metadatas = results.get("metadatas", [[]])[0] or []

    if not documents:
        results = documents_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=request.n_results,
        )
        documents = results.get("documents", [[]])[0] or []
        ids = results.get("ids", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []

    citations = [
        Citation(id=cid, document=doc, metadata=meta)
        for cid, doc, meta in zip(ids, documents, metadatas)
    ]

    context = "\n".join(documents) if documents else ""
    memory = get_memory(request.session_id)  # Uses MAX_MEMORY_ENTRIES by default

    if context:
        answer = answer_with_context(user_query, context, memory)
        grounding = "grounded"
        add_memory(
            request.session_id,
            user_query,
            answer,
            citations=[c.model_dump() for c in citations],
        )
    elif memory:
        answer = answer_with_memory(user_query, memory)
        grounding = "grounded"
        citations = latest_memory_citations(memory)
        add_memory(
            request.session_id,
            user_query,
            answer,
            citations=[c.model_dump() for c in citations],
        )
    else:
        answer = (
            "I don't know based on the uploaded documents. "
            "This looks like a general or ungrounded query."
        )
        grounding = "ungrounded"
        add_memory(request.session_id, user_query, answer, citations=[])

    return ChatResponse(
        intent=intent,
        user_query=user_query,
        filters=filters,
        grounding=grounding,
        answer=answer,
        citations=citations,
    )
