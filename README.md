# MOSDAC Help Bot - Backend (RAG System)

A powerful Retrieval-Augmented Generation (RAG) system built for the ISRO Hackathon that provides intelligent question-answering capabilities over MOSDAC satellite data documentation using local LLMs and geospatial search.

## 🚀 Features

- **Vector-based Semantic Search**: Uses FAISS and sentence transformers for efficient document retrieval
- **Local LLM Integration**: Powered by Ollama for privacy-preserving AI responses
- **Geospatial Queries**: Location-based document search with radius filtering
- **Multi-source Knowledge Base**: Ingests both PDF documents and web content
- **FastAPI Backend**: High-performance REST API with CORS support
- **Named Entity Recognition**: Extracts and geocodes location entities from documents
- **Interactive CLI**: Tools for building and managing the knowledge base

## 📋 Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running locally
- At least 4GB RAM for embedding models
- Internet connection for initial setup and geocoding

## 🛠️ Installation

1. **Clone the repository**
```bash
cd ISRO-Hackathon-Local-RAG-Bot
```

2. **Install Python dependencies**
```bash
pip install fastapi uvicorn sentence-transformers faiss-cpu numpy pydantic requests shapely haversine spacy folium
```

3. **Download spaCy language model**
```bash
python -m spacy download en_core_web_sm
```

4. **Install and start Ollama**
```bash
# Follow instructions at https://ollama.ai/
# Pull a model (e.g., llama3, mistral, or phi)
ollama pull llama3
```

5. **Set up environment variables** (optional)
Create a `.env` file if you need custom configurations.

## 📂 Project Structure

```
ISRO-Hackathon-Local-RAG-Bot/
├── app.py                      # FastAPI server with query endpoints
├── build_index.py              # Build FAISS vector index from documents
├── geospatial_pipeline.py      # Location extraction and geospatial queries
├── query_index_ollama.py       # Query logic with Ollama integration
├── load_chunks.py              # Document loading utilities
├── pdf_chunks/                 # PDF document chunks (2169 files)
├── web_chunks/                 # Web-scraped content chunks (100 files)
├── vector_store/               # FAISS index and metadata
│   ├── index.faiss            # Vector embeddings index
│   └── metadata.pkl           # Document metadata
├── geocache.pkl               # Cached geocoding results
└── .env                       # Environment configuration
```

## 🚦 Quick Start

### 1. Build the Vector Index

Process all documents and create the searchable vector database:

```bash
python build_index.py
```

This will:
- Load documents from `pdf_chunks/` and `web_chunks/`
- Generate embeddings using `all-MiniLM-L6-v2`
- Build a FAISS index for fast similarity search
- Save to `vector_store/`

### 2. (Optional) Enrich with Geospatial Data

Extract and geocode location entities from documents:

```bash
python geospatial_pipeline.py
```

Interactive menu options:
1. Extract location names from documents
2. Enrich with latitude/longitude coordinates
3. Save metadata
4. Query by distance from a location
5. Query by polygon (GeoJSON)
6. Visualize results on a map

### 3. Start the API Server

```bash
python app.py
```

The server will start at `http://localhost:8000`

## 🔌 API Endpoints

### Standard Query
**POST** `/query`

Ask questions about the knowledge base.

```json
{
  "question": "Which satellites provide ocean wind data?"
}
```

**Response:**
```json
{
  "answer": "Based on the documentation, SCATSAT-1 and Oceansat-2 provide ocean surface wind data..."
}
```

### Geospatial Query
**POST** `/query-geospatial`

Query documents related to a specific location.

```json
{
  "question": "What satellite data is available for this region?",
  "location": "Kerala",
  "radius_km": 100
}
```

**Response:**
```json
{
  "answer": "For the Kerala region, the following datasets are available..."
}
```

## 🧪 Testing the API

Using curl:
```bash
# Standard query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is INSAT-3D used for?"}'

# Geospatial query
curl -X POST http://localhost:8000/query-geospatial \
  -H "Content-Type: application/json" \
  -d '{"question": "Rainfall data near Mumbai", "location": "Mumbai", "radius_km": 50}'
```

## 🧠 How It Works

1. **Document Ingestion**: PDF and web content is chunked into manageable pieces
2. **Embedding Generation**: Each chunk is converted to a vector using sentence transformers
3. **Vector Storage**: FAISS index enables fast similarity search
4. **Query Processing**: User questions are embedded and matched against the index
5. **Context Retrieval**: Top-k most relevant chunks are retrieved
6. **Answer Generation**: Ollama LLM generates answers based on retrieved context
7. **Geospatial Enhancement**: Location entities are extracted and geocoded for spatial queries

## 🗺️ Geospatial Features

The system can:
- Extract location names from documents using NER
- Geocode locations using OpenStreetMap Nominatim API
- Find documents within a radius of a location
- Filter documents within a polygon boundary (GeoJSON)
- Visualize results on interactive maps

**Cache System**: Geocoding results are cached in `geocache.pkl` to avoid redundant API calls.

## ⚙️ Configuration

### Ollama Model Selection
Edit `query_index_ollama.py` to change the model:
```python
def choose_model():
    return "llama3"  # or "mistral", "phi", etc.
```

### Embedding Model
Edit `build_index.py` to use a different embedding model:
```python
model = SentenceTransformer("all-MiniLM-L6-v2")
# Alternatives: "all-mpnet-base-v2", "multi-qa-MiniLM-L6-cos-v1"
```

### Search Parameters
Adjust in `app.py`:
```python
results = retrieve_with_geo(question, k=5, max_score=1.0)
# k: number of results to retrieve
# max_score: maximum distance threshold
```

## 📊 Data Sources

- **PDF Chunks**: 2,169 text files extracted from MOSDAC documentation
- **Web Chunks**: 100 text files scraped from MOSDAC web portal
- **Total Knowledge Base**: ~2,269 document chunks

## 🔧 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'faiss'`
```bash
pip install faiss-cpu
```

**Issue**: Ollama connection error
```bash
# Check if Ollama is running
ollama list
# Start Ollama service if needed
```

**Issue**: Geocoding rate limits
- The system includes automatic retry logic and caching
- Requests are throttled to 1 per second
- Results are saved every 10 locations

**Issue**: Out of memory during indexing
- Process documents in smaller batches
- Use a smaller embedding model
- Increase system swap space

## 🚀 Performance Tips

- Use GPU-accelerated FAISS for large datasets (`faiss-gpu`)
- Increase `k` parameter for more comprehensive results
- Adjust chunk size in document processing for better granularity
- Use faster Ollama models like `phi` for quicker responses

## 📝 License

Built for ISRO Hackathon 2024

## 👥 Contributing

This is a hackathon project. Feel free to fork and extend!

## 🙏 Acknowledgments

- ISRO MOSDAC for satellite data and documentation
- Ollama for local LLM capabilities
- Sentence Transformers for embedding models
- FAISS for efficient vector search
