import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
