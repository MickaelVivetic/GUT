from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import tempfile
import os
import base64
import chromadb

from main import RAGAgent
from ingestion import TextIngestion
from config import CHROMA_DB_PATH, TEXT_COLLECTION_NAME
import database as db
from vision_extraction import extract_products_from_base64


agent: Optional[RAGAgent] = None
ingestion_instances: dict = {}


def get_ingestion(client_id: str) -> TextIngestion:
    if client_id not in ingestion_instances:
        ingestion_instances[client_id] = TextIngestion(client_id=client_id)
    return ingestion_instances[client_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    db.init_database()
    agent = RAGAgent()
    yield
    agent = None


app = FastAPI(
    title="GUT RAG API",
    description="API pour le systeme RAG avec LangChain et Ollama",
    version="1.0.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    question: str
    client_id: str
    top_k: int = 5


class QueryResponse(BaseModel):
    answer: str
    question: str
    client_id: str


class IngestTextRequest(BaseModel):
    text: str
    client_id: str
    source: str = "api"


class IngestResponse(BaseModel):
    message: str
    client_id: str
    chunks_count: int


class HealthResponse(BaseModel):
    status: str
    message: str


class ProductMetadata(BaseModel):
    titre_legende: Optional[str] = None
    description: Optional[str] = None
    plusP: Optional[str] = None
    prix_principal: Optional[Dict[str, str]] = None
    prix_barre: Optional[Dict[str, str]] = None
    reduction: Optional[str] = None
    image_path: Optional[str] = None


class ProductRequest(BaseModel):
    id_produit: str
    source_file: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProductUpdateRequest(BaseModel):
    source_file: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProductResponse(BaseModel):
    message: str
    product: Dict[str, Any]


class ClientResponse(BaseModel):
    client_id: str
    name: Optional[str]
    created_at: str


class VectorDBInfo(BaseModel):
    collection_name: str
    client_id: str
    documents_count: int


class ExtractedProduct(BaseModel):
    id_produit: str
    source_file: str
    content: str
    metadata: Dict[str, Any]


class VisionExtractionResponse(BaseModel):
    message: str
    client_id: str
    products_count: int
    products: List[ExtractedProduct]


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", message="Service is running")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        answer = agent.query(request.question, request.client_id, request.top_k)
        return QueryResponse(answer=answer, question=request.question, client_id=request.client_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(request: IngestTextRequest):
    try:
        ingestion = get_ingestion(request.client_id)
        chunks_count = ingestion.ingest_text(request.text, request.source)
        return IngestResponse(
            message="Text ingested successfully",
            client_id=request.client_id,
            chunks_count=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(client_id: str, file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        ingestion = get_ingestion(client_id)
        chunks_count = ingestion.ingest_file(tmp_path)
        
        os.unlink(tmp_path)
        
        return IngestResponse(
            message=f"File '{file.filename}' ingested successfully",
            client_id=client_id,
            chunks_count=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/extract/image", response_model=VisionExtractionResponse)
async def extract_products_from_image_route(
    client_id: str,
    file: UploadFile = File(...),
    save_to_db: bool = True
):
    allowed_extensions = [".jpg", ".jpeg", ".png", ".webp"]
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Format non supporte. Utilisez: {allowed_extensions}")
    
    try:
        content = await file.read()
        image_base64 = base64.b64encode(content).decode("utf-8")
        
        products = extract_products_from_base64(image_base64)
        
        if not products:
            return VisionExtractionResponse(
                message="Aucun produit extrait",
                client_id=client_id,
                products_count=0,
                products=[]
            )
        
        for p in products:
            p["source_file"] = file.filename
        
        if save_to_db:
            ingestion = get_ingestion(client_id)
            for product in products:
                try:
                    db.create_product(client_id, product)
                    if product.get("content"):
                        ingestion.ingest_text(product["content"], source=f"product_{product['id_produit']}")
                except Exception:
                    pass
        
        return VisionExtractionResponse(
            message=f"{len(products)} produits extraits de '{file.filename}'",
            client_id=client_id,
            products_count=len(products),
            products=[ExtractedProduct(**p) for p in products]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/pdf", response_model=VisionExtractionResponse)
async def extract_products_from_pdf_route(
    client_id: str,
    file: UploadFile = File(...),
    save_to_db: bool = True
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Format non supporte. Utilisez un fichier PDF.")
    
    try:
        from pdf2image import convert_from_bytes
        
        content = await file.read()
        images = convert_from_bytes(content, dpi=150)
        
        all_products = []
        ingestion = get_ingestion(client_id) if save_to_db else None
        
        for page_num, image in enumerate(images, 1):
            import io
            img_buffer = io.BytesIO()
            image.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            
            page_products = extract_products_from_base64(img_base64)
            
            for p in page_products:
                p["source_file"] = f"{file.filename}_page_{page_num}"
                p["id_produit"] = f"{p['id_produit']}_p{page_num}"
                
                if save_to_db:
                    try:
                        db.create_product(client_id, p)
                        if p.get("content"):
                            ingestion.ingest_text(p["content"], source=f"product_{p['id_produit']}")
                    except Exception:
                        pass
                
                all_products.append(p)
        
        return VisionExtractionResponse(
            message=f"{len(all_products)} produits extraits de {len(images)} pages",
            client_id=client_id,
            products_count=len(all_products),
            products=[ExtractedProduct(**p) for p in all_products]
        )
    except ImportError:
        raise HTTPException(status_code=500, detail="pdf2image non installe. Installez avec: pip install pdf2image")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "GUT RAG API", "docs": "/docs"}


@app.get("/clients", response_model=List[ClientResponse])
async def list_clients():
    try:
        clients = db.list_clients()
        return [
            ClientResponse(
                client_id=c["client_id"],
                name=c.get("name"),
                created_at=str(c["created_at"])
            ) for c in clients
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clients/{client_id}")
async def create_client(client_id: str, name: Optional[str] = None):
    try:
        client = db.create_client(client_id, name)
        return {"message": "Client created", "client": client}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vector-databases", response_model=List[VectorDBInfo])
async def list_vector_databases():
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collections = chroma_client.list_collections()
        
        result = []
        for col in collections:
            if col.name.startswith(TEXT_COLLECTION_NAME + "_"):
                client_id = col.name.replace(TEXT_COLLECTION_NAME + "_", "")
                result.append(VectorDBInfo(
                    collection_name=col.name,
                    client_id=client_id,
                    documents_count=col.count()
                ))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/products/{client_id}", response_model=ProductResponse)
async def create_product(client_id: str, product: ProductRequest):
    try:
        product_data = product.model_dump()
        created = db.create_product(client_id, product_data)
        
        if product.content:
            ingestion = get_ingestion(client_id)
            ingestion.ingest_text(product.content, source=f"product_{product.id_produit}")
        
        return ProductResponse(message="Product created", product=created)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/products/{client_id}/{id_produit}", response_model=ProductResponse)
async def update_product(client_id: str, id_produit: str, product: ProductUpdateRequest):
    try:
        existing = db.get_product(client_id, id_produit)
        if not existing:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product_data = product.model_dump(exclude_none=True)
        updated = db.update_product(client_id, id_produit, product_data)
        
        if product.content:
            ingestion = get_ingestion(client_id)
            try:
                ingestion.collection.delete(where={"source": f"product_{id_produit}"})
            except:
                pass
            ingestion.ingest_text(product.content, source=f"product_{id_produit}")
        
        return ProductResponse(message="Product updated", product=updated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{client_id}/{id_produit}")
async def get_product(client_id: str, id_produit: str):
    try:
        product = db.get_product(client_id, id_produit)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/products/{client_id}")
async def list_products(client_id: str, limit: int = 100, offset: int = 0):
    try:
        products = db.list_products(client_id, limit, offset)
        return {"client_id": client_id, "products": products, "count": len(products)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/products/{client_id}/{id_produit}")
async def delete_product(client_id: str, id_produit: str):
    try:
        ingestion = get_ingestion(client_id)
        try:
            ingestion.collection.delete(where={"source": f"product_{id_produit}"})
        except:
            pass
        
        deleted = db.delete_product(client_id, id_produit)
        if not deleted:
            raise HTTPException(status_code=404, detail="Product not found")
        return {"message": "Product deleted", "id_produit": id_produit}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/products/{client_id}/bulk")
async def bulk_create_products(client_id: str, products: List[ProductRequest]):
    try:
        products_data = [p.model_dump() for p in products]
        count = db.bulk_upsert_products(client_id, products_data)
        
        ingestion = get_ingestion(client_id)
        for p in products:
            if p.content:
                ingestion.ingest_text(p.content, source=f"product_{p.id_produit}")
        
        return {"message": f"{count} products created/updated", "client_id": client_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
