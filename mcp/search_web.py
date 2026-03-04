import logging
import os
import sys

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from llm import LLM  # provider-agnostic

mcp = FastMCP("WebSearch")


def _tavily_search(query: str) -> str:
    payload = {"query": query, "search_depth": "basic", "include_answers": True, "max_results": 5}
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return "No search results found."
        return "\n\n".join(f"{r.get('title', '')}\n{r.get('content', '')}" for r in results)
    except Exception as e:
        logging.error(f"Tavily search error: {e}")
        return f"An error occurred during search: {e}"


@mcp.tool()
async def search_web(query: str) -> str:
    logging.info(f"Search request: {query}")
    content = _tavily_search(query)
    message = await LLM.ainvoke(
        f"Summarize the following search results in one paragraph:\n\n{content}"
    )
    return message.content


if __name__ == "__main__":
    mcp.run(transport="stdio")
