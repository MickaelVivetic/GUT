import base64
import json
import re
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import OLLAMA_BASE_URL, OLLAMA_VISON


EXTRACTION_PROMPT = """Analyse cette image de catalogue/magazine commercial.
Extrais TOUS les produits visibles un par un.

Pour chaque produit, retourne un objet JSON avec:
- id_produit: un identifiant unique (utilise le nom du produit en snake_case)
- titre: nom complet du produit
- marque: marque du produit si visible
- description: description du produit
- prix_actuel: prix actuel (nombre uniquement)
- prix_barre: ancien prix barre si visible (nombre uniquement)
- reduction: pourcentage ou montant de reduction si visible
- categorie: categorie du produit (ex: vetement, equipement, accessoire)
- public_cible: adulte, enfant, ou les deux

Retourne UNIQUEMENT un tableau JSON valide, sans texte avant ou apres.
Exemple: [{"id_produit": "casque_ski_atomic", "titre": "Casque de ski", ...}, ...]"""


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_products_from_image(image_path: str) -> List[Dict[str, Any]]:
    image_base64 = encode_image_to_base64(image_path)
    
    payload = {
        "model": OLLAMA_VISON,
        "prompt": EXTRACTION_PROMPT,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    
    result = response.json()
    raw_response = result.get("response", "")
    
    return parse_json_response(raw_response)


def extract_products_from_base64(image_base64: str) -> List[Dict[str, Any]]:
    payload = {
        "model": OLLAMA_VISON,
        "prompt": EXTRACTION_PROMPT,
        "images": [image_base64],
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    
    result = response.json()
    raw_response = result.get("response", "")
    
    return parse_json_response(raw_response)


def parse_json_response(raw_response: str) -> List[Dict[str, Any]]:
    json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
    if json_match:
        try:
            products = json.loads(json_match.group())
            return normalize_products(products)
        except json.JSONDecodeError:
            pass
    
    try:
        products = json.loads(raw_response)
        if isinstance(products, list):
            return normalize_products(products)
        return [normalize_product(products)]
    except json.JSONDecodeError:
        return []


def normalize_product(product: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "id_produit": product.get("id_produit", "unknown"),
        "source_file": "vision_extraction",
        "content": build_content_text(product),
        "metadata": {
            "titre_legende": product.get("titre", ""),
            "description": product.get("description", ""),
            "marque": product.get("marque", ""),
            "prix_principal": {
                "FLC": str(product.get("prix_actuel", "")),
                "PrixDetailsFLc": ""
            },
            "prix_barre": {
                "FLC": str(product.get("prix_barre", "")),
                "PrixDetailsFLc": ""
            },
            "reduction": str(product.get("reduction", "")),
            "categorie": product.get("categorie", ""),
            "public_cible": product.get("public_cible", ""),
            "image_path": ""
        }
    }
    return normalized


def normalize_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [normalize_product(p) for p in products]


def build_content_text(product: Dict[str, Any]) -> str:
    parts = []
    
    if product.get("titre"):
        parts.append(f"Produit: {product['titre']}")
    
    if product.get("marque"):
        parts.append(f"Marque: {product['marque']}")
    
    if product.get("description"):
        parts.append(f"Description: {product['description']}")
    
    if product.get("prix_actuel"):
        parts.append(f"Prix: {product['prix_actuel']} EUR")
    
    if product.get("prix_barre"):
        parts.append(f"Ancien prix: {product['prix_barre']} EUR")
    
    if product.get("reduction"):
        parts.append(f"Reduction: {product['reduction']}")
    
    if product.get("categorie"):
        parts.append(f"Categorie: {product['categorie']}")
    
    if product.get("public_cible"):
        parts.append(f"Pour: {product['public_cible']}")
    
    return ". ".join(parts)
