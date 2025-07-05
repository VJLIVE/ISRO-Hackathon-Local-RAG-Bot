import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(folder):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, encoding='utf-8') as f:
                text = f.read()
                docs.append({'content': text, 'source': path})
    return docs

print("📄 Loading documents...")
all_docs = []
all_docs += load_chunks("web_chunks")
all_docs += load_chunks("pdf_chunks")
print(f"✅ Loaded {len(all_docs)} documents.")

print("🤖 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [doc['content'] for doc in all_docs]
sources = [doc['source'] for doc in all_docs]

print("🔷 Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

print("📦 Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs("vector_store", exist_ok=True)

faiss.write_index(index, "vector_store/index.faiss")
with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(all_docs, f)

print("\n✅ Done! Vector index and metadata saved in ./vector_store")
