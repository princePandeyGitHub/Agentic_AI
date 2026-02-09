from fastapi import FastAPI

from backend.api.chat import router as chat_router
from backend.api.ingest import router as ingest_router

app = FastAPI(title="Agentic RAG System")

app.include_router(ingest_router)
app.include_router(chat_router)


@app.get("/")
def health():
    return {"status": "ok"}
