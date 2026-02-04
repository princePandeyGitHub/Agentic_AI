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

# =========================
# 3️⃣ Verify counts
# =========================
print("✅ HR collection count:", hr_collection.count())
print("✅ Incident collection count:", incident_collection.count())

# =========================
# 4️⃣ Quick semantic query function
# =========================
def query_collection(collection, query_text, n_results=1):

    query_embedding = model.encode(query_text)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results["documents"][0]


# =========================
# 5️⃣ Test queries
# =========================
hr_query = "Can you tell me about adams"
incident_query = "why my internet is too slow"

hr_result = query_collection(hr_collection, hr_query)
incident_result = query_collection(incident_collection, incident_query)

print("\n--- HR Query Result ---")
print(hr_result[0])  # top HR document

print("\n--- Incident Query Result ---")
print(incident_result[0])  # top Incident document

# one more check
query_embedding = model.encode("What is maternity leave policy")

res = hr_collection.query(
    query_embeddings=[query_embedding.tolist()],
    n_results=3
)

for i in range(3):
    print("\nRank", i + 1)
    print("Doc:", res["documents"][0][i])
    print("Meta:", res["metadatas"][0][i])
