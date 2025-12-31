from typing import List, Dict, Any, Optional
from PIL import Image
import open_clip
import torch
import chromadb
from langchain_ollama import OllamaEmbeddings

from config import (
    CHROMA_DB_PATH, TEXT_EMBEDDING_MODEL, OLLAMA_BASE_URL,
    TEXT_COLLECTION_NAME, IMAGE_COLLECTION_NAME,
    CLIP_MODEL, CLIP_PRETRAINED
)


class TextRetriever:
    def __init__(self, client_id: str = "default"):
        self.client_id = client_id
        self.embeddings = OllamaEmbeddings(
            model=TEXT_EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        self._init_chroma()

    def _init_chroma(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection_name = f"{TEXT_COLLECTION_NAME}_{self.client_id}"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                documents.append({
                    "text": doc,
                    "source": results["metadatas"][0][i].get("source", "unknown"),
                    "score": 1 - results["distances"][0][i]
                })
        
        return documents

    def get_context(self, query: str, top_k: int = 5) -> str:
        docs = self.search(query, top_k)
        context = "\n\n".join([doc["text"] for doc in docs])
        return context


class ImageRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.model = self.model.to(self.device)
        self.model.eval()
        self._init_chroma()

    def _init_chroma(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.collection = self.client.get_or_create_collection(
            name=IMAGE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        text_tokens = self.tokenizer([text]).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_text(text_tokens)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze().cpu().tolist()

    def _get_image_embedding(self, image_path: str) -> List[float]:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        
        return embedding.squeeze().cpu().tolist()

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self._get_text_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        images = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, meta in enumerate(results["metadatas"][0]):
                images.append({
                    "image_path": meta.get("image_path", ""),
                    "description": meta.get("description", ""),
                    "score": 1 - results["distances"][0][i]
                })
        
        return images

    def search_by_image(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self._get_image_embedding(image_path)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        images = []
        if results["metadatas"] and results["metadatas"][0]:
            for i, meta in enumerate(results["metadatas"][0]):
                images.append({
                    "image_path": meta.get("image_path", ""),
                    "description": meta.get("description", ""),
                    "score": 1 - results["distances"][0][i]
                })
        
        return images


class MultiModalRetriever:
    def __init__(self):
        self.text_retriever = TextRetriever()
        self.image_retriever = ImageRetriever()

    def search(self, query: str, top_k: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        return {
            "texts": self.text_retriever.search(query, top_k),
            "images": self.image_retriever.search_by_text(query, top_k)
        }
