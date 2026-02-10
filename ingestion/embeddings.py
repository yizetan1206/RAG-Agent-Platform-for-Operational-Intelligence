from typing import List
import time
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """
    SentenceTransformer-based embedding provider.
    Swappable later with OpenAI / cloud embeddings.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name

        start_time = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        load_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "Embedding model loaded",
            extra={
                "model": self.model_name,
                "load_ms": load_ms,
            }
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        logger.debug(
            "Embedding batch started",
            extra={"text_count": len(texts)}
        )

        start_time = time.perf_counter()

        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception:
            logger.exception("Embedding batch failed")
            raise

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "Embedding batch completed",
            extra={
                "text_count": len(texts),
                "latency_ms": latency_ms,
            }
        )

        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        logger.debug("Embedding query started")

        start_time = time.perf_counter()

        try:
            embedding = self.model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception:
            logger.exception("Query embedding failed")
            raise

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.debug(
            "Embedding query completed",
            extra={"latency_ms": latency_ms}
        )

        return embedding[0].tolist()
