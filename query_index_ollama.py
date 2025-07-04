import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama
import spacy

print("🤖 Loading model & index…")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("vector_store/index.faiss")

with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

nlp = spacy.load("en_core_web_sm")

print(f"✅ Loaded {len(metadata)} documents.")

def retrieve_with_geo(query, k=5, max_score=1.0):
    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k*3)
    doc = nlp(query)
    query_locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    results = []
    for i, dist in zip(I[0], D[0]):
        result = metadata[i]
        result['score'] = dist
        if dist > max_score:
            continue
        if query_locations:
            if any(loc.lower() in [l.lower() for l in result.get("locations", [])] for loc in query_locations):
                results.append(result)
        else:
            results.append(result)

        if len(results) >= k:
            break
    return results

def choose_model():
    result = ollama.list()
    models = result.get('models', [])
    if not models:
        print("❌ No models found. Please run `ollama pull <model>` to download one.")
        exit(1)
    print("\nAvailable models:")
    for i, m in enumerate(models):
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
    ollama_model = choose_model()
    while True:
        user_query = input("\n❓ Enter your query (or type 'exit' to quit):\n> ")
        if user_query.lower() in ["exit", "quit"]:
            break
        results = retrieve_with_geo(user_query, k=5, max_score=1.0)
        if not results:
            print("⚠️ No good match found in the knowledge base.")
            continue
        context = "\n\n".join(
            [f"Source: {r['source']}\nContent: {r['content'][:1000]}" for r in results]
        )
        print("\n📝 Relevant Context:\n")
        print(context[:2000])
        print("\n🤖 Generating answer with Ollama…")
        answer = generate_answer(user_query, context, ollama_model)
        print("\n💬 Answer:\n")
        print(answer)
