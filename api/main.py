from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

from api.health import router as health_router
from api.schemas.query import QueryRequest, QueryResponse
from ingestion.loader import DocumentLoader
from ingestion.chunker import chunk_text
from ingestion.embeddings import EmbeddingProvider
from vectorstore.faiss_store import FAISSStore
from core.rag_service import RAGService
from core.llm_provider import LLMProvider

from dotenv import load_dotenv
load_dotenv()

from core.logger import setup_logger

setup_logger(
    level="INFO",
    service_name="ai-knowledge-assistant"
)
import logging

logger = logging.getLogger(__name__)



INDEX_PATH = Path("data/faiss.index")
META_PATH = Path("data/faiss_meta.pkl")


def init_rag_service() -> RAGService:
    embedder = EmbeddingProvider()

    if INDEX_PATH.exists() and META_PATH.exists():
        store = FAISSStore.load(str(INDEX_PATH), str(META_PATH))
        logger.info("FAISS index loaded from disk")
    else:
        loader = DocumentLoader("tests/datasets/test_docs")
        docs = loader.load()

        chunks = []
        metadatas = []
        for doc in docs:
            for chunk in chunk_text(doc["content"]):
                chunks.append(chunk)
                metadatas.append({
                    "source": doc["source"],
                    "text": chunk
                })

        embeddings = embedder.embed_texts(chunks)

        store = FAISSStore(dim=len(embeddings[0]))
        store.add_vectors(embeddings, metadatas)

        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        store.save(str(INDEX_PATH), str(META_PATH))
        logger.info("FAISS index built and saved")


    llm = LLMProvider()

    return RAGService(
        store=store,
        embedder=embedder,
        llm=llm
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup initiated")

    try:
        app.state.rag_service = init_rag_service()
        logger.info("RAG service initialized successfully")
        yield
    except Exception:
        logger.exception("Failed during application startup")
        raise
    finally:
        logger.info("Application shutdown completed")



app = FastAPI(
    title="AI Knowledge Assistant",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(health_router)


@app.get("/")
def root():
    return {"status": "ok", "service": "ai-knowledge-assistant"}


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    rag_service: RAGService = app.state.rag_service

    logger.debug(
        "Received query",
        extra={"top_k": request.top_k}
    )

    result = rag_service.query(
        request.question,
        request.top_k
    )

    logger.info("Query processed successfully")

    return {
        "question": request.question,
        "answer": result["answer"],
        "contexts": result["contexts"],
    }
