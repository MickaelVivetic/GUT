import psycopg2
from psycopg2.extras import RealDictCursor, Json
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from config import POSTGRES_CONFIG


@contextmanager
def get_db_connection():
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clients (
                    client_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    id SERIAL PRIMARY KEY,
                    id_produit VARCHAR(255) NOT NULL,
                    client_id VARCHAR(255) NOT NULL,
                    source_file VARCHAR(255),
                    content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(id_produit, client_id),
                    FOREIGN KEY (client_id) REFERENCES clients(client_id) ON DELETE CASCADE
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_products_client ON products(client_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_products_id_produit ON products(id_produit)
            """)
            conn.commit()


def create_client(client_id: str, name: str = None) -> Dict[str, Any]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO clients (client_id, name)
                VALUES (%s, %s)
                ON CONFLICT (client_id) DO UPDATE SET name = EXCLUDED.name
                RETURNING *
            """, (client_id, name or client_id))
            conn.commit()
            return dict(cur.fetchone())


def list_clients() -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM clients ORDER BY created_at")
            return [dict(row) for row in cur.fetchall()]


def create_product(client_id: str, product_data: Dict[str, Any]) -> Dict[str, Any]:
    create_client(client_id)
    
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                INSERT INTO products (id_produit, client_id, source_file, content, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id_produit, client_id) DO UPDATE SET
                    source_file = EXCLUDED.source_file,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
                RETURNING *
            """, (
                product_data["id_produit"],
                client_id,
                product_data.get("source_file"),
                product_data.get("content"),
                Json(product_data.get("metadata", {}))
            ))
            conn.commit()
            return dict(cur.fetchone())


def get_product(client_id: str, id_produit: str) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM products WHERE client_id = %s AND id_produit = %s
            """, (client_id, id_produit))
            row = cur.fetchone()
            return dict(row) if row else None


def update_product(client_id: str, id_produit: str, product_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                UPDATE products SET
                    source_file = COALESCE(%s, source_file),
                    content = COALESCE(%s, content),
                    metadata = COALESCE(%s, metadata),
                    updated_at = CURRENT_TIMESTAMP
                WHERE client_id = %s AND id_produit = %s
                RETURNING *
            """, (
                product_data.get("source_file"),
                product_data.get("content"),
                Json(product_data.get("metadata")) if product_data.get("metadata") else None,
                client_id,
                id_produit
            ))
            conn.commit()
            row = cur.fetchone()
            return dict(row) if row else None


def delete_product(client_id: str, id_produit: str) -> bool:
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM products WHERE client_id = %s AND id_produit = %s
            """, (client_id, id_produit))
            conn.commit()
            return cur.rowcount > 0


def list_products(client_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM products WHERE client_id = %s
                ORDER BY updated_at DESC
                LIMIT %s OFFSET %s
            """, (client_id, limit, offset))
            return [dict(row) for row in cur.fetchall()]


def bulk_upsert_products(client_id: str, products: List[Dict[str, Any]]) -> int:
    create_client(client_id)
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            for product in products:
                cur.execute("""
                    INSERT INTO products (id_produit, client_id, source_file, content, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id_produit, client_id) DO UPDATE SET
                        source_file = EXCLUDED.source_file,
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    product["id_produit"],
                    client_id,
                    product.get("source_file"),
                    product.get("content"),
                    Json(product.get("metadata", {}))
                ))
            conn.commit()
            return len(products)
