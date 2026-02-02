from ingestion.embeddings import EmbeddingProvider
from vectorstore.faiss_store import FAISSStore
from ingestion.chunker import chunk_text


class RAGService:
    def __init__(self, store: FAISSStore, embedder: EmbeddingProvider):
        self.store = store
        self.embedder = embedder

    def query(self, question: str, top_k: int = 5):
        query_vec = self.embedder.embed_query(question)
        return self.store.query(query_vec, top_k)