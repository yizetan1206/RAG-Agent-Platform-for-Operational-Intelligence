from typing import List
import os
import time
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMProvider:
    """
    Thin wrapper around LLM API.
    Swappable later (OpenAI / Azure / local).
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        logger.info(
            "LLMProvider initialized",
            extra={"model": self.model}
        )

    def generate(self, question: str, contexts: List[str]) -> str:
        start_time = time.perf_counter()

        # NOTE: Do NOT log prompt, question, or contexts
        logger.debug(
            "LLM generation started",
            extra={"context_count": len(contexts)}
        )

        context_block = "\n\n".join(contexts)

        prompt = f"""
You are a helpful AI assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context_block}

Question:
{question}

Answer:
""".strip()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
        except Exception:
            logger.exception("LLM API call failed")
            raise

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        logger.info(
            "LLM generation completed",
            extra={
                "model": self.model,
                "latency_ms": latency_ms,
            }
        )

        return response.choices[0].message.content.strip()
