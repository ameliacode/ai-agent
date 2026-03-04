from langgraph.prebuilt import create_react_agent  # noqa: F401 (deprecated import path, still functional in V1.x)

from llm import LLM
from tools import TOOLS

agent = create_react_agent(LLM, TOOLS)


def run(query: str) -> str:
    result = agent.invoke({"messages": [("user", query)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    print("Research Agent ready. Type 'exit' to quit.\n")
    while True:
        query = input("Query: ").strip()
        if query.lower() == "exit":
            break
        print("\n" + run(query) + "\n")
