import json
import chromadb
from sentence_transformers import SentenceTransformer

# load embedding model - mixedbread-ai
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# create chromadb client
client = chromadb.PersistentClient(path="chroma_db")

# create collections
hr_collection = client.get_or_create_collection(name="hr_data")
incident_collection = client.get_or_create_collection(name="incident_data")

# load both the json data
with open("data/hr_data.json", "r", encoding="utf-8") as f:
    hr_json = json.load(f)

with open("data/incident_data.json", "r", encoding="utf-8") as f:
    incident_json = json.load(f)

# we have two types of data in both hr and incident

# in hr we have employees -> hr_records and policies -> hr_policies
hr_records = hr_json["records"]
hr_policies = hr_json["policies"]

#in incident we have incidents -> incident_records and trouble shooting guidelines -> incident_guides
incident_records = incident_json["incidents"]
incident_guides = incident_json["troubleshooting_guides"]

# =========================
# Helper: convert record → text
# =========================
def record_to_text(record: dict):
    parts = []
    for k, v in record.items():
        if isinstance(v, list):
            v = ", ".join(map(str, v))
        parts.append(f"{k}: {v}")
    return " | ".join(parts)

# =========================
# Ingest HR records (batch embedding)
# =========================
hr_texts = [record_to_text(r) for r in hr_records]
hr_embeddings = model.encode(hr_texts, batch_size=16, show_progress_bar=True)

for idx, record in enumerate(hr_records):
    hr_collection.add(
        ids=[record["employee_id"]],
        documents=[hr_texts[idx]],
        embeddings=[hr_embeddings[idx].tolist()],
        metadatas={
            "type": "hr",
            "department": record.get("department"),
            "employee_id": record.get("employee_id")
        }
    )

# ingest hr policies
policy_texts = [record_to_text(p) for p in hr_policies]
policy_embeddings = model.encode(policy_texts, batch_size=16, show_progress_bar=True)

for idx, policy in enumerate(hr_policies):
    hr_collection.add(
        ids=[policy["policy_id"]],
        documents=[policy_texts[idx]],
        embeddings=[policy_embeddings[idx].tolist()],
        metadatas={
            "type": "policy",
            "category": policy.get("category"),
            "policy_id": policy.get("policy_id")
        }
    )


# =========================
# Ingest Incident record data (batch embedding)
# =========================
incident_texts = [record_to_text(r) for r in incident_records]
incident_embeddings = model.encode(incident_texts, batch_size=16, show_progress_bar=True)

for idx, record in enumerate(incident_records):
    incident_collection.add(
        ids=[record["incident_id"]],
        documents=[incident_texts[idx]],
        embeddings=[incident_embeddings[idx].tolist()],
        metadatas={
            "type": "incident",
            "category": record.get("category"),
            "severity": record.get("severity"),
            "incident_id": record.get("incident_id")
        }
    )

# ingest incident guide data
guide_texts = [record_to_text(g) for g in incident_guides]
guide_embeddings = model.encode(guide_texts, batch_size=16, show_progress_bar=True)

for idx, guide in enumerate(incident_guides):
    incident_collection.add(
        ids=[guide["guide_id"]],
        documents=[guide_texts[idx]],
        embeddings=[guide_embeddings[idx].tolist()],
        metadatas={
            "type": "troubleshooting",
            "category": guide.get("category"),
            "guide_id": guide.get("guide_id")
        }
    )

# =========================
# 8️⃣ Verify ingestion
# =========================
print("✅ Ingestion complete")
print("HR count:", hr_collection.count())
print("Incident count:", incident_collection.count())
