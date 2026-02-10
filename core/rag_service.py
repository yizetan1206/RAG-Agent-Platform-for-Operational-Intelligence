import logging
from ingestion.embeddings import EmbeddingProvider
from vectorstore.faiss_store import FAISSStore
from ingestion.chunker import chunk_text
from core.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        store: FAISSStore,
        embedder: EmbeddingProvider,
        llm: LLMProvider,
    ):
        self.store = store
        self.embedder = embedder
        self.llm = llm

        logger.info("RAGService initialized")

    def query(self, question: str, top_k: int = 5):
        logger.debug("RAG query started", extra={"top_k": top_k})

        # ---- Embedding ----
        try:
            query_vec = self.embedder.embed_query(question)
        except Exception:
            logger.exception("Failed to generate query embedding")
            raise

        # ---- Retrieval ----
        retrieved = self.store.query(query_vec, top_k)
        logger.debug(
            "Vector retrieval completed",
            extra={"retrieved_count": len(retrieved)}
        )

        # ---- Hard guard: similarity threshold ----
        filtered = [
            r for r in retrieved
            if r.get("score", 0.0) > 0.5
        ]

        if not filtered:
            logger.warning(
                "Query rejected due to low similarity",
                extra={
                    "top_k": top_k,
                    "max_score": max(
                        (r.get("score", 0.0) for r in retrieved),
                        default=0.0,
                    ),
                },
            )
            return {
                "answer": "I don't have enough reliable information to answer that.",
                "contexts": [],
            }

        logger.info(
            "Relevant contexts found",
            extra={
                "context_count": len(filtered),
                "max_score": max(r["score"] for r in filtered),
            },
        )

        # ---- Generation ----
        contexts = [r["text"] for r in filtered]

        try:
            answer = self.llm.generate(question, contexts)
        except Exception:
            logger.exception("LLM generation failed")
            raise

        logger.debug("LLM generation completed successfully")

        return {
            "answer": answer,
            "contexts": filtered,
        }
