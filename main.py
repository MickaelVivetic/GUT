from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import OLLAMA_BASE_URL, OLLAMA_MODEL
from ingestion import TextIngestion, ImageIngestion
from retrieval import TextRetriever, ImageRetriever, MultiModalRetriever


class RAGAgent:
    def __init__(self, model: str = None):
        self.llm = ChatOllama(
            model=model or OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0
        )
        self.text_retrievers = {}
        self.image_retrievers = {}
        self._setup_chain()

    def _setup_chain(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un assistant IA qui repond aux questions en utilisant le contexte fourni.
Utilise uniquement les informations du contexte pour repondre. Si tu ne trouves pas la reponse dans le contexte, dis-le clairement.

Contexte:
{context}"""),
            ("human", "{question}")
        ])
        
        self.chain = (
            {"context": lambda x: x["context"], "question": lambda x: x["question"]}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def get_text_retriever(self, client_id: str = "default") -> TextRetriever:
        if client_id not in self.text_retrievers:
            self.text_retrievers[client_id] = TextRetriever(client_id=client_id)
        return self.text_retrievers[client_id]

    def query(self, question: str, client_id: str = "default", top_k: int = 5) -> str:
        text_retriever = self.get_text_retriever(client_id)
        context = text_retriever.get_context(question, top_k)
        
        response = self.chain.invoke({
            "context": context,
            # "context": "Realisation d'agent RAG",
            "question": question
        })
        
        return response

    # def query_with_sources(self, question: str, top_k: int = 5) -> dict:
    #     if self.text_retriever is None:
    #         self.initialize_retrievers()
        
    #     docs = self.text_retriever.search(question, top_k)
    #     context = "\n\n".join([doc["text"] for doc in docs])
        
    #     response = self.chain.invoke({
    #         "context": context,
    #         "question": question
    #     })
        
    #     return {
    #         "answer": response,
    #         "sources": [{"source": doc["source"], "score": doc["score"]} for doc in docs]
    #     }

    def search_images(self, query: str, top_k: int = 5) -> list:
        if self.image_retriever is None:
            self.initialize_retrievers()
        
        return self.image_retriever.search_by_text(query, top_k)


def ingest_documents(file_path: Optional[str] = None, directory_path: Optional[str] = None):
    ingestion = TextIngestion()
    
    if file_path:
        count = ingestion.ingest_file(file_path)
        print(f"Ingested {count} chunks from {file_path}")
    
    if directory_path:
        count = ingestion.ingest_directory(directory_path)
        print(f"Ingested {count} chunks from {directory_path}")


def ingest_images(directory_path: str):
    ingestion = ImageIngestion()
    count = ingestion.ingest_directory(directory_path)
    print(f"Ingested {count} images from {directory_path}")


if __name__ == "__main__":
    agent = RAGAgent()
    
    print("RAG Agent initialized.")
    
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == "/quit":
            break
        
        if user_input.startswith("/ingest_images "):
            directory = user_input[15:].strip()
            ingest_images(directory)
            continue
        
        if user_input.startswith("/ingest "):
            file_path = user_input[8:].strip()
            ingest_documents(file_path=file_path)
            continue
        
        # if user_input.startswith("/search_images "):
        #     query = user_input[15:].strip()
        #     results = agent.search_images(query)
        #     for i, img in enumerate(results, 1):
        #         print(f"  {i}. {img['image_path']} (score: {img['score']:.3f})")
        #     continue
        
        # result = agent.query_with_sources(user_input)
        result = agent.query(user_input)
        print(f"\nAssistant: {result}")
        print()
