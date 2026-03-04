import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import MODEL

_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise EnvironmentError("OPENAI_API_KEY is not set. Add it to your .env file.")

LLM = ChatOpenAI(model=MODEL, api_key=_api_key)


def ask(prompt: str, max_tokens: int = 600) -> str:
    return LLM.invoke([HumanMessage(content=prompt)], max_tokens=max_tokens).content
