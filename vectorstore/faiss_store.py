import faiss
import numpy as np
from typing import List, Dict
import pickle


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
    
    # ---------------- Persistence Methods ----------------

    def save(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata separately."""
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadatas, f)

    @classmethod
    def load(cls, index_path: str, metadata_path: str):
        """Load FAISS index and metadata into a new FAISSStore object."""
        index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            metadatas = pickle.load(f)

        store = cls(dim=index.d)
        store.index = index
        store.metadatas = metadatas
        return store