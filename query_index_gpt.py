import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

# 🔷 Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or "your-api-key-here"

client = OpenAI(api_key=api_key)

# 🔷 Load embedding model & FAISS index
print("🤖 Loading model & index…")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("vector_store/index.faiss")

with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"✅ Loaded {len(metadata)} documents.")

# 🔷 Function to retrieve relevant chunks
def retrieve(query, k=3, max_score=1.0):
    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k)

    results = []
    for i, dist in zip(I[0], D[0]):
        result = metadata[i]
        result['score'] = dist
        if dist <= max_score:
            results.append(result)
    return results

# 🔷 Function to generate GPT answer
def generate_answer(question, context):
    prompt = f"""You are a helpful assistant for the MOSDAC portal. 
Answer the question based only on the following context. 
If the answer is not explicitly stated, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    while True:
        user_query = input("\n❓ Enter your query (or type 'exit' to quit):\n> ")
        if user_query.lower() in ["exit", "quit"]:
            break

        results = retrieve(user_query, k=5, max_score=1.0)

        if not results:
            print("⚠️ No good match found in the knowledge base.")
            continue

        context = "\n\n".join([
            f"Source: {r['source']}\nContent: {r['content'][:1000]}"
            for r in results
        ])
        print("\n📝 Relevant Context:\n")
        print(context[:2000])  # optional: show retrieved context
        print("\n🤖 Generating GPT answer…")

        answer = generate_answer(user_query, context)
        print("\n💬 GPT Answer:\n")
        print(answer)
