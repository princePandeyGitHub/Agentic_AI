import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# load the model to encode the query also
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1",)

# --------------------------
# 1️⃣ Open ChromaDB
# --------------------------
client_chroma = chromadb.PersistentClient(path="chroma_db")

hr_collection = client_chroma.get_collection(name="hr_data")
incident_collection = client_chroma.get_collection(name="incident_data")

# --------------------------
# 2️⃣ Initialize Groq LLaMA
# --------------------------
client_groq = Groq(api_key=GROQ_API_KEY)

# --------------------------
# 3️⃣ Helper: query Chroma
# --------------------------
def query_collection(collection, query_text, n_results=1):

    query_embedding = model.encode(query_text)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results["documents"][0]

# --------------------------
# 4️⃣ Automatic collection detection
# --------------------------
def detect_collection(query):
    hr_keywords = ["employee", "salary", "bonus", "leave", "benefits", "department", "role", "join date"]
    it_keywords = ["laptop", "printer", "network", "internet", "hardware", "software", "slow", "incident"]

    query_lower = query.lower()

    if any(word in query_lower for word in hr_keywords):
        return "hr"
    elif any(word in query_lower for word in it_keywords):
        return "incident"
    else:
        return "both"  # fallback to both collections

# --------------------------
# 5️⃣ LLaMA wrapper
# --------------------------
def ask_ai(query, context):
    prompt = f"""
You are a helpful assistant. Use ONLY the context below to answer. you should also perform reasoning if required
If the answer is not in the context, say "I don't know."

Context:
{context}

User question:
{query}

Answer concisely:
"""
    response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=256
    )
    return response.choices[0].message.content

# --------------------------
# 6️⃣ Interactive loop
# --------------------------
if __name__ == "__main__":
    print("=== Agentic AI System ===")
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        collection_type = detect_collection(query)
        context_docs = []

        if collection_type in ("hr", "both"):
            hr_docs = query_collection(hr_collection, query, n_results=3)
            context_docs.extend(hr_docs)

        if collection_type in ("incident", "both"):
            incident_docs = query_collection(incident_collection, query, n_results=3)
            context_docs.extend(incident_docs)

        context = "\n".join(context_docs) if context_docs else "No relevant context found."
        print(context)

        answer = ask_ai(query, context)
        print("\nAI Answer:\n", answer)
