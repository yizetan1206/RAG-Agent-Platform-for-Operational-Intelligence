import faiss
import numpy as np
from typing import List, Dict


class FAISSStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas: List[Dict] = []

    def add_vectors(self, vectors: List[List[float]], metadatas: List[Dict]):
        vectors_np = np.array(vectors, dtype="float32")
        self.index.add(vectors_np)
        self.metadatas.extend(metadatas)

    def query(self, vector: List[float], top_k: int = 5):
        vector_np = np.array([vector], dtype="float32")
        scores, indices = self.index.search(vector_np, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.metadatas):
                results.append({
                    "score": float(score),
                    **self.metadatas[idx],
                })
        return results
