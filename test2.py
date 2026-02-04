import chromadb
from sentence_transformers import SentenceTransformer

# load the model to encode the query also
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1",)

# =========================
# 1️⃣ Open persistent ChromaDB
# =========================
client = chromadb.PersistentClient(path="chroma_db")

# =========================
# 2️⃣ Load collections
# =========================
hr_collection = client.get_collection(name="hr_data")
incident_collection = client.get_collection(name="incident_data")

query_text = "can you tell me about adams"
query_embedding = model.encode(query_text)

hr_res = hr_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=1
)

print(hr_res["documents"])

incident_res = incident_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=1
)

print(incident_res["documents"])