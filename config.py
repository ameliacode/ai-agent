import os
import certifi
from dotenv import load_dotenv

load_dotenv()

# Fix broken SSL_CERT_FILE often set by conda environments
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

MODEL = os.getenv("MODEL", "gpt-4o-mini")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.getenv("DOCS_DIR", os.path.join(BASE_DIR, "docs"))
OBSIDIAN_DIR = os.getenv("OBSIDIAN_DIR", os.path.join(BASE_DIR, "obsidian_export"))
FILE_SEARCH_ROOT = os.getenv("FILE_SEARCH_ROOT", os.path.expanduser("~"))
