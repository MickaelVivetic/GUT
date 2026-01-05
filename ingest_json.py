import json
import argparse
import sys
from typing import List, Dict, Any

from database import create_client, bulk_upsert_products, create_product
from ingestion import TextIngestion


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        return [data]
    return data


def ingest_products(client_id: str, products: List[Dict[str, Any]], verbose: bool = True):
    create_client(client_id, name=client_id)
    
    if verbose:
        print(f"Client: {client_id}")
        print(f"Produits a ingerer: {len(products)}")
        print("-" * 40)
    
    ingestion = TextIngestion(client_id=client_id)
    
    pg_count = 0
    vector_count = 0
    
    for product in products:
        id_produit = product.get("id_produit")
        if not id_produit:
            id_produit = product.get("source_file", "unknown").replace(".html", "")
            product["id_produit"] = id_produit
        
        try:
            create_product(client_id, product)
            pg_count += 1
        except Exception as e:
            if verbose:
                print(f"  Erreur PostgreSQL pour {id_produit}: {e}")
        
        content = product.get("content", "")
        if content:
            try:
                ingestion.ingest_text(content, source=f"product_{id_produit}")
                vector_count += 1
            except Exception as e:
                if verbose:
                    print(f"  Erreur ChromaDB pour {id_produit}: {e}")
        
        if verbose:
            titre = product.get("metadata", {}).get("titre_legende", "")[:40]
            print(f"  {id_produit}: {titre}...")
    
    if verbose:
        print("-" * 40)
        print(f"PostgreSQL: {pg_count} produits")
        print(f"ChromaDB: {vector_count} documents")
    
    return {"postgres": pg_count, "chroma": vector_count}


def main():
    parser = argparse.ArgumentParser(description="Ingestion de produits JSON vers PostgreSQL et ChromaDB")
    parser.add_argument("--client", "-c", required=True, help="ID du client")
    parser.add_argument("--file", "-f", default="data.json", help="Fichier JSON source (default: data.json)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Mode silencieux")
    
    args = parser.parse_args()
    
    try:
        products = load_json_file(args.file)
    except FileNotFoundError:
        print(f"Erreur: Fichier '{args.file}' non trouve")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Erreur: JSON invalide - {e}")
        sys.exit(1)
    
    result = ingest_products(args.client, products, verbose=not args.quiet)
    
    if not args.quiet:
        print(f"\nIngestion terminee pour le client '{args.client}'")


if __name__ == "__main__":
    main()
