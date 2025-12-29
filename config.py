import os
from dotenv import load_dotenv

load_dotenv()


POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": int(os.getenv("POSTGRES_PORT", 5432)),
    "database": os.getenv("POSTGRES_DB", "rag_db"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_data")

TEXT_EMBEDDING_MODEL = "nomic-embed-text"
TEXT_EMBEDDING_DIM = 768

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
IMAGE_EMBEDDING_DIM = 512

TEXT_COLLECTION_NAME = "text_embeddings"
IMAGE_COLLECTION_NAME = "image_embeddings"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:8b"
