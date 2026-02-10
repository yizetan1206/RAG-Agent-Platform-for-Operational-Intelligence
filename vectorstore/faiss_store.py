import faiss
import numpy as np
from typing import List, Dict
import pickle
import logging
import os

logger = logging.getLogger(__name__)


class FAISSStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[Dict] = []

        logger.info(
            "FAISSStore initialized",
            extra={"dimension": dim}
        )

    def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict]):
        if len(vectors) != len(metadatas):
            logger.error(
                "Vector/metadata count mismatch",
                extra={
                    "vector_count": len(vectors),
                    "metadata_count": len(metadatas),
                },
            )
            raise ValueError("Vectors and metadatas length mismatch")

        vectors_np = np.array(vectors, dtype="float32")

        logger.info(
            "Adding vectors to FAISS index",
            extra={"count": vectors_np.shape[0]}
        )

        try:
            self.index.add(vectors_np)
        except Exception:
            logger.exception("Failed to add vectors to FAISS index")
            raise

        self.metadatas.extend(metadatas)

        logger.debug(
            "Vectors added successfully",
            extra={
                "total_vectors": self.index.ntotal,
            }
        )

    def query(self, vector: List[float], top_k: int = 5):
        logger.debug(
            "FAISS query started",
            extra={
                "top_k": top_k,
                "index_size": self.index.ntotal,
            }
        )

        vector_np = np.array([vector], dtype="float32")

        try:
            scores, indices = self.index.search(vector_np, top_k)
        except Exception:
            logger.exception("FAISS search failed")
            raise

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadatas):
                results.append(
                    {
                        "score": float(score),
                        **self.metadatas[idx],
                    }
                )

        logger.debug(
            "FAISS query completed",
            extra={
                "returned": len(results),
                "top_score": max((r["score"] for r in results), default=0.0),
            }
        )

        return results

    # ---------------- Persistence Methods ----------------

    def save(self, index_path: str, metadata_path: str):
        logger.info(
            "Saving FAISS index to disk",
            extra={
                "index_path": index_path,
                "metadata_path": metadata_path,
                "vector_count": self.index.ntotal,
            }
        )

        try:
            faiss.write_index(self.index, index_path)
            with open(metadata_path, "wb") as f:
                pickle.dump(self.metadatas, f)
        except Exception:
            logger.exception("Failed to save FAISS index or metadata")
            raise

        logger.info("FAISS index saved successfully")

    @classmethod
    def load(cls, index_path: str, metadata_path: str):
        logger.info(
            "Loading FAISS index from disk",
            extra={
                "index_path": index_path,
                "metadata_path": metadata_path,
            }
        )

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.error("FAISS index or metadata file missing")
            raise FileNotFoundError("FAISS index or metadata file missing")

        try:
            index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                metadatas = pickle.load(f)
        except Exception:
            logger.exception("Failed to load FAISS index or metadata")
            raise

        store = cls(dim=index.d)
        store.index = index
        store.metadatas = metadatas

        logger.info(
            "FAISS index loaded",
            extra={
                "dimension": index.d,
                "vector_count": index.ntotal,
            }
        )

        return store
