import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama

# 🔷 Load embedding model & index
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


# 🔷 Function to let you choose an installed Ollama model
def choose_model():
    result = ollama.list()
    models = result.get('models', [])

    if not models:
        print("❌ No models found. Please run `ollama pull <model>` to download one.")
        exit(1)

    print("\nAvailable models:")
    for i, m in enumerate(models):
        # Each m is likely a dict with keys like 'model' and maybe 'details'
        print(f"{i+1}. {m.get('model', 'unknown')}")

    while True:
        try:
            choice = int(input("\nSelect model number to use: "))
            if 1 <= choice <= len(models):
                return models[choice-1].get('model', None)
            else:
                print("❌ Invalid choice. Try again.")
        except ValueError:
            print("❌ Please enter a number.")

# 🔷 Function to generate answer using Ollama
def generate_answer(question, context, model_name):
    prompt = f"""You are a helpful assistant for the MOSDAC portal.
Answer the question based only on the following context.
If the answer is not explicitly stated, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content'].strip()


if __name__ == "__main__":
    # choose which ollama model to use
    ollama_model = choose_model()

    while True:
        user_query = input("\n❓ Enter your query (or type 'exit' to quit):\n> ")
        if user_query.lower() in ["exit", "quit"]:
            break

        results = retrieve(user_query, k=5, max_score=1.0)

        if not results:
            print("⚠️ No good match found in the knowledge base.")
            continue

        context = "\n\n".join(
            [f"Source: {r['source']}\nContent: {r['content'][:1000]}" for r in results]
        )
        print("\n📝 Relevant Context:\n")
        print(context[:2000])  # optional: show retrieved context
        print("\n🤖 Generating answer with Ollama…")

        answer = generate_answer(user_query, context, ollama_model)
        print("\n💬 Answer:\n")
        print(answer)
