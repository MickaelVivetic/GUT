import sys
from retrieval import TextRetriever


def test_search(query: str = None):
    print("Initialisation de TextRetriever...")
    retriever = TextRetriever()
    
    if query:
        queries = [query]
    else:
        print("\nYou:")
        queries = []
        while True:
            q = input("\nQuestion: ").strip()
            if q.lower() == 'quit':
                break
            if q:
                queries.append(q)
                
                results = retriever.search(q, top_k=5)
                
                if not results:
                    print("Aucun resultat trouve.")
                    continue
                    
                for i, doc in enumerate(results, 1):
                    print(f"\n--- Resultat {i} (score: {doc['score']:.4f}) ---")
                    print(f"Source: {doc['source']}")
                    print(f"Texte: {doc['text'][:500]}...")
        return
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Question: {query}")
        print('='*60)
        
        results = retriever.search(query, top_k=5)
        
        if not results:
            print("Aucun resultat trouve.")
            continue
            
        for i, doc in enumerate(results, 1):
            print(f"\n--- Resultat {i} (score: {doc['score']:.4f}) ---")
            print(f"Source: {doc['source']}")
            print(f"Texte: {doc['text'][:500]}...")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    test_search(query)
