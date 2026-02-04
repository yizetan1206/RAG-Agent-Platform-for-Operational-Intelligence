from ingestion.embeddings import EmbeddingProvider
from vectorstore.faiss_store import FAISSStore
from ingestion.chunker import chunk_text
from core.llm_provider import LLMProvider


class RAGService:
    def __init__(self, store: FAISSStore, embedder: EmbeddingProvider, llm: LLMProvider):
        self.store = store
        self.embedder = embedder
        self.llm = llm

    def query(self, question: str, top_k: int = 5):
        query_vec = self.embedder.embed_query(question)
        retrieved = self.store.query(query_vec, top_k)

        contexts = [item["text"] for item in retrieved]
        answer = self.llm.generate(question, contexts)


        return {
            "answer": answer,
            "contexts": retrieved,
        }