from fastapi import FastAPI
from api.health import router as health_router
from api.schemas.query import QueryRequest, QueryResponse
from ingestion.loader import DocumentLoader
from ingestion.chunker import chunk_text
from ingestion.embeddings import EmbeddingProvider
from vectorstore.faiss_store import FAISSStore
from core.rag_service import RAGService
from core.llm_provider import LLMProvider


app = FastAPI(
    title="AI Knowledge Assistant",
    version="0.1.0"
)

app.include_router(health_router)


@app.get("/")
def root():
    return {"status": "ok", "service": "ai-knowledge-assistant"}


# ---------- RAG BOOTSTRAP ----------
loader = DocumentLoader("tests/datasets/test_docs")
docs = loader.load()
print(f"Loaded {len(docs)} documents",flush=True)

chunks = []
metadatas = []

for doc in docs:
    doc_chunks = chunk_text(doc["content"])
    for chunk in doc_chunks:
        chunks.append(chunk)
        metadatas.append({
            "source": doc["source"],
            "text": chunk,
        })


embedder = EmbeddingProvider()
embeddings = embedder.embed_texts(chunks)

store = FAISSStore(dim=len(embeddings[0]))
store.add_vectors(embeddings, metadatas)
# ----------------------------------

llm = LLMProvider()

rag_service = RAGService(
    store=store,
    embedder=embedder,
    llm=llm
)



@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    result = rag_service.query(request.question, request.top_k)
    return {
        "question": request.question,
        "answer": result["answer"],
        "contexts": result["contexts"],
    }
