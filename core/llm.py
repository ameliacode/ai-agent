from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from config import MODEL

# Base LLM — shared across all calls (stateless, thread-safe)
LLM = ChatOllama(model=MODEL)


def ask(prompt: str, max_tokens: int = 600) -> str:
    """Single-turn LLM call. max_tokens caps output length."""
    return LLM.bind(num_predict=max_tokens).invoke(
        [HumanMessage(content=prompt)]
    ).content
