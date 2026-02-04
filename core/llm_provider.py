from typing import List
import os
from openai import OpenAI


class LLMProvider:
    """
    Thin wrapper around LLM API.
    Swappable later (OpenAI / Azure / local).
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, question: str, contexts: List[str]) -> str:
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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        return response.choices[0].message.content.strip()
