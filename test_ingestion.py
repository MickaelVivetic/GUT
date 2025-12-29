import sys
import time
from ingestion import TextIngestion

PDF_PATH = "C://Users//16777//Downloads//Cahier.pdf"


def test_pdf_ingestion(pdf_path: str = None):
    if pdf_path is None:
        pdf_path = PDF_PATH
    
    print(f"Fichier PDF: {pdf_path}")
    print("Initialisation de TextIngestion...")
    
    start_time = time.time()
    ingestion = TextIngestion()
    
    print("Ingestion du PDF en cours ...")
    num_chunks = ingestion.ingest_file(pdf_path)
    
    elapsed = time.time() - start_time
    print(f"\nIngestion terminee!")
    print(f"Nombre de chunks crees: {num_chunks}")
    print(f"Temps ecoule: {elapsed:.2f} secondes")


if __name__ == "__main__":
    pdf_file = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    test_pdf_ingestion(pdf_file)
