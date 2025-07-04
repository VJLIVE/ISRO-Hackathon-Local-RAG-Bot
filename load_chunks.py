import os

def load_chunks(folder):
    docs = []
    for fname in os.listdir(folder):
        if fname.endswith(".txt"):
            path = os.path.join(folder, fname)
            with open(path, encoding='utf-8') as f:
                text = f.read()
                docs.append({'content': text, 'source': path})
    return docs

if __name__ == "__main__":
    all_docs = []
    all_docs += load_chunks("web_chunks")
    all_docs += load_chunks("pdf_chunks")

    print(f"✅ Loaded {len(all_docs)} documents.")
    print("Sample document:")
    print(f"Source: {all_docs[0]['source']}")
    print(f"Content:\n{all_docs[0]['content'][:500]}…")
