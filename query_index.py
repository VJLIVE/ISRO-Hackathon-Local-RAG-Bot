import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# 🔷 Load embedding model, FAISS index & metadata
print("🤖 Loading model & index…")
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("vector_store/index.faiss")

with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"✅ Loaded {len(metadata)} documents.")

# 🔷 Search function
def search(query, k=3, max_score=1.0):
    print(f"\n🔎 Query: {query}")
    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k)

    results = []
    for i, dist in zip(I[0], D[0]):
        result = metadata[i]
        result['score'] = dist
        results.append(result)

    # Filter out results worse than threshold
    filtered = [r for r in results if r['score'] <= max_score]
    return filtered, results

if __name__ == "__main__":
    while True:
        user_query = input("\n❓ Enter your query (or type 'exit' to quit):\n> ")
        if user_query.lower() in ["exit", "quit"]:
            break

        filtered_results, all_results = search(user_query, k=5, max_score=1.0)

        if not filtered_results:
            print("⚠️ No good match found for your query. (Best score was {:.2f})".format(all_results[0]['score']))
            continue

        for idx, r in enumerate(filtered_results, 1):
            print(f"\n--- Result {idx} ---")
            print(f"📄 Source: {r['source']}")
            print(f"💬 Content:\n{r['content'][:1000]}…")
            print(f"🔗 Score: {r['score']:.4f}")
