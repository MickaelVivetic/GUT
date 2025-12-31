import os
from typing import List, Optional
from PIL import Image
import open_clip
import torch
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader

from config import (
    CHROMA_DB_PATH, TEXT_EMBEDDING_MODEL, OLLAMA_BASE_URL,
    IMAGE_EMBEDDING_DIM, TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME,
    CHUNK_SIZE, CHUNK_OVERLAP, CLIP_MODEL, CLIP_PRETRAINED
)


class TextIngestion:
    def __init__(self, client_id: str = "default"):
        self.client_id = client_id
        self.embeddings = OllamaEmbeddings(
            model=TEXT_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        self._init_chroma()

    def _init_chroma(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection_name = f"{TEXT_COLLECTION_NAME}_{self.client_id}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_text(self, text: str, source: str = "manual"):
        chunks = self.text_splitter.split_text(text)
        embeddings = self.embeddings.embed_documents(chunks)
        
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": source}] * len(chunks)
        )
        return len(chunks)

    def ingest_file(self, file_path: str):
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        base_id = os.path.basename(file_path).replace(".", "_")
        ids = [f"{base_id}_{i}" for i in range(len(texts))]
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": file_path}] * len(texts)
        )
        return len(chunks)

    def ingest_directory(self, directory_path: str, glob: str = "**/*.txt"):
        loader = DirectoryLoader(directory_path, glob=glob)
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        texts = [doc.page_content for doc in chunks]
        sources = [doc.metadata.get("source", "unknown") for doc in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        ids = [f"dir_{i}" for i in range(len(texts))]
        metadatas = [{"source": src} for src in sources]
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        return len(chunks)


class ImageIngestion:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._init_chroma()

    def _init_chroma(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=IMAGE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_image_embedding(self, image_path: str) -> List[float]:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze().cpu().tolist()

    def ingest_image(self, image_path: str, description: str = ""):
        embedding = self._get_image_embedding(image_path)
        
        img_id = os.path.basename(image_path).replace(".", "_")
        self.collection.add(
            ids=[img_id],
            embeddings=[embedding],
            metadatas=[{"image_path": image_path, "description": description}]
        )
        return 1

    def ingest_directory(self, directory_path: str, extensions: List[str] = None):
        if extensions is None:
            extensions = [".jpg", ".jpeg", ".png", ".webp"]
        
        count = 0
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    self.ingest_image(file_path)
                    count += 1
        
        return count
