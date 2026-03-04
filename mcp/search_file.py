import asyncio
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

logging.basicConfig(level=logging.INFO)
mcp = FastMCP("File-Search")

ROOT_DIR = os.getenv("FILE_SEARCH_ROOT", os.path.expanduser("~"))


def _search_files(keyword: str, base_path: str = ROOT_DIR, max_results: int = 20) -> list[dict]:
    results = []
    for dirpath, _, filenames in os.walk(base_path):
        for fname in filenames:
            if keyword.lower() in fname.lower():
                fpath = os.path.abspath(os.path.join(dirpath, fname))
                try:
                    stat = os.stat(fpath)
                    results.append({
                        "filename": fname,
                        "path": fpath,
                        "size(bytes)": stat.st_size,
                        "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    if len(results) >= max_results:
                        return results
                except Exception as e:
                    logging.warning(f"File access error: {fpath} - {e}")
    return results


@mcp.tool()
async def find_files(keyword: str) -> str:
    logging.info(f"Starting file search for keyword: {keyword}")
    loop = asyncio.get_event_loop()
    found = await loop.run_in_executor(None, _search_files, keyword)
    if not found:
        return f"No files found for keyword: {keyword}"
    return "\n".join(
        f"{r['filename']} - {r['path']} - {r['size(bytes)']} bytes - {r['created']}"
        for r in found
    )


if __name__ == "__main__":
    asyncio.run(mcp.run(transport="stdio"))
