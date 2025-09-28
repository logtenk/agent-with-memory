import os
from dotenv import load_dotenv
load_dotenv()

LLAMACPP_BASE_URL = os.getenv("LLAMACPP_BASE_URL", "http://127.0.0.1:8000")
CHROMA_PERSIST_ROOT = os.getenv("CHROMA_PERSIST_ROOT", "./data/agents")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
