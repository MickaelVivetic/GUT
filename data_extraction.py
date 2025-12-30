import json
import re
import os
from bs4 import BeautifulSoup

# --- CONFIGURATION ---
DOSSIER_HTML = "./html"  # Nom du dossier contenant tes fichiers .html
FICHIER_SORTIE = "data.json" # Nom du fichier JSON final

# --- 1. FONCTIONS D'EXTRACTION (Identiques à avant) ---

def clean_text(text):
    if not text:
        return ""
    return " ".join(text.replace('\xa0', ' ').split())

def extract_data_from_html(html_content, filename):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    container = soup.select_one('.WeldomProd24Detaille')
    
    if not container:
        # On retourne None si la structure n'est pas trouvée
        return None

    # -- Titre --
    titre_elem = container.select_one('.titreLegende span')
    titre_legende = clean_text(titre_elem.text) if titre_elem else "Titre inconnu"

    # -- Description --
    legende_div = container.select_one('.legende')
    description = ""
    if legende_div:
        titre_p = legende_div.find('p', class_='titreLegende')
        if titre_p:
            desc_p = titre_p.find_next_sibling('p')
            if desc_p:
                description = clean_text(desc_p.text)

    # -- PlusP --
    plus_p_elem = container.select_one('.plusP')
    plus_p = clean_text(plus_p_elem.text) if plus_p_elem else ""

    # -- Image --
    image_div = container.select_one('.Packshot-Principal')
    image_path = ""
    if image_div and 'style' in image_div.attrs:
        style = image_div['style']
        match = re.search(r"url\('([^']+)'\)", style)
        if match:
            image_path = match.group(1)

    # -- Prix Principal --
    prix_principal_data = {"FLC": "", "PrixDetailsFLc": ""}
    prix_princ_div = container.select_one('.PrixPrincipal')
    if prix_princ_div:
        flc = prix_princ_div.select_one('.FLC')
        details = prix_princ_div.select_one('.PrixDetails')
        prix_principal_data["FLC"] = clean_text(flc.text) if flc else ""
        prix_principal_data["PrixDetailsFLc"] = clean_text(details.text) if details else ""

    # -- Prix Barré --
    prix_barre_data = {"FLC": "", "PrixDetailsFLc": ""}
    prix_barre_div = container.select_one('.PrixBarre')
    if prix_barre_div:
        flc = prix_barre_div.select_one('.FLC')
        details = prix_barre_div.select_one('.PrixDetails')
        prix_barre_data["FLC"] = clean_text(flc.text) if flc else ""
        prix_barre_data["PrixDetailsFLc"] = clean_text(details.text) if details else ""

    # -- Réduction --
    reduc_elem = container.select_one('.PrixReduc')
    reduction = clean_text(reduc_elem.text) if reduc_elem else ""

    # -- Construction du texte "content" --
    full_price = f"{prix_principal_data['FLC']} {prix_principal_data['PrixDetailsFLc']}"
    content_text = (
        f"Description textuelle : {titre_legende}. "
        f"{description} "
        f"Info: {plus_p}. "
        f"Prix: {full_price}"
    )

    # -- Objet Final --
    donnee_produit = {
        "source_file": filename, # Ajout utile pour savoir de quel fichier ça vient
        "content": content_text,
        "metadata": {
            "titre_legende": titre_legende,
            "description": description,
            "plusP": plus_p,
            "prix_principal": prix_principal_data,
            "prix_barre": prix_barre_data,
            "reduction": reduction,
            "image_path": image_path
        }
    }
    
    return donnee_produit

# --- 2. FONCTION PRINCIPALE DE TRAITEMENT DE DOSSIER ---

def traiter_dossier(dossier_entree, fichier_sortie):
    # Liste pour stocker tous les produits trouvés
    tous_les_produits = []
    
    # Vérification si le dossier existe
    if not os.path.exists(dossier_entree):
        print(f"ERREUR : Le dossier '{dossier_entree}' n'existe pas.")
        # On crée le dossier pour l'utilisateur pour la prochaine fois
        os.makedirs(dossier_entree)
        print(f"Je viens de créer le dossier '{dossier_entree}'. Mets tes fichiers HTML dedans et relance le script.")
        return

    print(f"--- Démarrage du traitement dans : {dossier_entree} ---")

    # On liste les fichiers du dossier
    fichiers = [f for f in os.listdir(dossier_entree) if f.endswith('.html')]
    
    if not fichiers:
        print("Aucun fichier .html trouvé dans le dossier.")
        return

    compteur = 0

    for fichier in fichiers:
        chemin_complet = os.path.join(dossier_entree, fichier)
        
        try:
            # Lecture
            with open(chemin_complet, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Extraction
            donnees = extract_data_from_html(html_content, fichier)
            
            if donnees:
                tous_les_produits.append(donnees)
                print(f" Succès : {fichier} -> {donnees['metadata']['titre_legende'][:30]}...")
                compteur += 1
            else:
                print(f" Ignoré : {fichier} (Structure HTML non reconnue)")
                
        except Exception as e:
            print(f"Erreur sur le fichier {fichier} : {e}")

    # --- Sauvegarde finale ---
    if tous_les_produits:
        with open(fichier_sortie, 'w', encoding='utf-8') as f:
            json.dump(tous_les_produits, f, indent=4, ensure_ascii=False)
        
        print("-" * 40)
        print(f"TERMINÉ ! {compteur} produits extraits.")
        print(f"Fichier JSON sauvegardé : {fichier_sortie}")
        print("-" * 40)
    else:
        print("Aucune donnée n'a été extraite.")

# --- 3. LANCEMENT ---

if __name__ == "__main__":
    traiter_dossier(DOSSIER_HTML, FICHIER_SORTIE)