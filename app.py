from fastapi import FastAPI
from pydantic import BaseModel
from query_index_ollama import retrieve_with_geo, generate_answer, choose_model
from geospatial_pipeline import find_nearby_chunks
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

ollama_model = None  # global

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ollama_model
    print("🚀 Starting up…")
    ollama_model = choose_model()
    yield
    print("🛑 Shutting down…")

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

class GeoQuery(BaseModel):
    question: str
    location: str
    radius_km: float = 100

@app.post("/query")
async def query(data: Query):
    question = data.question
    results = retrieve_with_geo(question, k=5, max_score=1.0)

    if not results:
        return {"answer": "⚠️ No good match found in the knowledge base."}

    context = "\n\n".join(
        [f"Source: {r['source']}\nContent: {r['content'][:1000]}" for r in results]
    )

    answer = generate_answer(question, context, ollama_model)

    return {"answer": answer}

@app.post("/query-geospatial")
async def query_geospatial(data: GeoQuery):
    question = data.question
    location = data.location
    radius = data.radius_km

    results = find_nearby_chunks(location, radius)
    if not results:
        return {"answer": f"⚠️ No results found near {location} within {radius} km."}

    # use the retrieved context to generate an answer
    context = "\n\n".join(
        [f"Source: {entry['source']}\nContent: {entry['content'][:1000]}" for entry, loc, dist in results[:5]]
    )

    answer = generate_answer(question, context, ollama_model)

    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
