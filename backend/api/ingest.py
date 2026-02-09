import json
import uuid
from io import BytesIO
from typing import Dict, List, Tuple

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document as DocxDocument

from backend.core import documents_collection, model

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestResponse(BaseModel):
    status: str
    chunks_added: int
    ids: List[str]


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(end - overlap, 0)
    return chunks


def extract_text(file_bytes: bytes, filename: str) -> Tuple[str, str]:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, "pdf"
    if lower.endswith(".docx"):
        doc = DocxDocument(BytesIO(file_bytes))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        return text, "docx"
    if lower.endswith(".json"):
        parsed = json.loads(file_bytes.decode("utf-8"))
        text = json.dumps(parsed, ensure_ascii=True, indent=2)
        return text, "json"
    text = file_bytes.decode("utf-8", errors="ignore")
    return text, "text"


def build_metadata(intent: str, source_filename: str, source_type: str, extra: Dict) -> Dict:
    metadata = {
        "intent": intent,
        "source_filename": source_filename,
        "source_type": source_type,
    }
    if extra:
        for key, value in extra.items():
            metadata[key] = value
    return metadata


@router.post("/", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    intent: str = Form("general"),
    metadata_json: str = Form(""),
):
    file_bytes = await file.read()
    text, source_type = extract_text(file_bytes, file.filename or "upload")
    chunks = chunk_text(text)
    if not chunks:
        return IngestResponse(status="empty_document", chunks_added=0, ids=[])

    extra = {}
    if metadata_json:
        try:
            extra = json.loads(metadata_json)
        except json.JSONDecodeError:
            extra = {"metadata_parse_error": "invalid_json"}

    metadatas = []
    ids = []
    for idx, _ in enumerate(chunks):
        ids.append(str(uuid.uuid4()))
        metadatas.append(
            build_metadata(
                intent=intent,
                source_filename=file.filename or "upload",
                source_type=source_type,
                extra={"chunk_index": idx, **extra},
            )
        )

    embeddings = model.encode(chunks, batch_size=16, show_progress_bar=False)

    documents_collection.add(
        ids=ids,
        documents=chunks,
        embeddings=[vector.tolist() for vector in embeddings],
        metadatas=metadatas,
    )

    return IngestResponse(status="ok", chunks_added=len(chunks), ids=ids)
