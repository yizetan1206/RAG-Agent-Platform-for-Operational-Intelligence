from pydantic import BaseModel
from typing import List


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class RetrievedContext(BaseModel):
    source: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: List[RetrievedContext]
