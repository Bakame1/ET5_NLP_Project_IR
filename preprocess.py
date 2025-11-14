import spacy
import re
import string
import os
import pickle
from tqdm import tqdm

# Configuration des stop words et du modèle spaCy
# Charger le modèle français de spaCy si non présent, le télécharger
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    print("Modèle 'fr_core_news_sm' non trouvé. Téléchargement en cours...")
    spacy.cli.download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

################################## REGEX ##################################
# Expressions régulières pour identifier les bandeaux et sections non pertinentes
# Liste déduite manuellement à partir de certains des fichiers txt
# Regex multi-lignes
MULTI_LINE_REGEX = [
    re.compile(r"Cette section est vide(, insuffisamment détaillée ou incomplète)?.*?Votre aide est la bienvenue\s*!", re.DOTALL | re.IGNORECASE),
    re.compile(r"Cette section est vide.*?Votre aide est la bienvenue\s*!\s*Comment faire\s*\?", re.DOTALL | re.IGNORECASE),
    re.compile(r"Cet article doit être recyclé.*?section à recycler\s*\}\}", re.DOTALL | re.IGNORECASE),
    re.compile(r"Cet article ne cite pas suffisamment ses sources.*?Comment ajouter mes sources\s*\?", re.DOTALL | re.IGNORECASE),
    re.compile(r"Cet article.*?doit être recyclé.*?\{\{ section à recycler \}\}\s*\.", re.DOTALL | re.IGNORECASE),
]

# Regex ligne par ligne
SINGLE_LINE_REGEX = [
    re.compile(r"Cette section est vide .+?\.", re.IGNORECASE),
    re.compile(r"Votre aide est la bienvenue .+?\.", re.IGNORECASE),
    re.compile(r"Vous lisez un « bon article » labellisé en .+?\.", re.DOTALL | re.IGNORECASE),
    re.compile(r"Une réorganisation et une clarification du contenu paraissent nécessaires\s*\.", re.IGNORECASE),
    re.compile(r"Cet article.*?doit être recyclé.*?\(\s*\w+\s*\d+\s*\)\s*\.", re.DOTALL | re.IGNORECASE),
    re.compile(r"Améliorez-le , discutez des points à améliorer ou précisez les sections à recycler en utilisant { { section à recycler } } .", re.DOTALL | re.IGNORECASE),
]

################################## Stop words francais ##################################
STOP_WORDS = spacy.lang.fr.stop_words.STOP_WORDS

###################### Pretraitement phrases wikipedia ######################
def remove_wiki_noise(text):
    """Supprime les bandeaux, sections non pertinentes et autres bruits Wikipedia."""
    # D'abord appliquer les regex multi-lignes sur le texte complet
    cleaned_text = text
    for pattern in MULTI_LINE_REGEX:
        cleaned_text = pattern.sub("", cleaned_text)
    # Puis appliquer les regex ligne par ligne
    lines = cleaned_text.splitlines()
    cleaned_lines = []
    for line in lines:
        temp_line = line
        is_noise = False
        for pattern in SINGLE_LINE_REGEX:
            if pattern.search(temp_line):
                # Si la regex correspond à toute la ligne (ou presque), on la supprime
                if pattern.fullmatch(temp_line.strip()):
                   is_noise = True
                   break
                else:
                    # Sinon, on supprime juste la partie correspondante
                    temp_line = pattern.sub("", temp_line)
        # Garder la ligne si elle n'a pas été marquée comme bruit et n'est pas vide/blanche après nettoyage
        if not is_noise and temp_line.strip():
            cleaned_lines.append(temp_line)
    return "\n".join(cleaned_lines)

###################### Pretraitement sur 1 texte ######################
def preprocess_text(text):
    """
    Applique le pipeline de prétraitement complet sur un texte donné.
    1. Supprime le bruit Wikipedia (bandeaux, sections inutiles...).
    2. Met en minuscule.
    3. Tokenise et lemmatise avec spaCy.
    4. Supprime la ponctuation, les nombres et les stop words.
    """
    # 1. Supprimer le bruit Wikipedia
    cleaned_text = remove_wiki_noise(text)
    # 2. Mise en minuscule
    cleaned_text = cleaned_text.lower()
    # 3. Traitement spaCy (tokenisation, lemmatisation)
    doc = nlp(cleaned_text)
    # 4. Filtrage des tokens et lemmatisation
    processed_tokens = []
    for token in doc:
        # Vérifier si ce n'est PAS : ponctuation, nombre, stop word, espace/saut de ligne
        # Et si le token n'est pas trop court (ex: 1 caractère, souvent du bruit résiduel)
        if (not token.is_punct and
            not token.is_space and
            not token.like_num and
            token.lemma_ not in STOP_WORDS and # Vérifier le lemme contre les stop words
            len(token.text) > 1): # ignorer les tokens d'un seul caractère
            # 5. Récupérer le lemme (forme de base du mot)
            processed_tokens.append(token.lemma_)
    return processed_tokens

############# Préparation des données pour l'indexation #############
def prepare_data_for_indexing(directory_path, output_file="preprocessed_data.pkl"):
    """
    Prépare les données pour l'indexation en appliquant le prétraitement à chaque fichier dans le répertoire.
    Stocke les résultats dans un fichier pickle pour une utilisation future.
    """
    # Vérifier si le fichier de sortie existe déjà
    if os.path.exists(output_file):
        print(f"Chargement des données prétraitées depuis {output_file}...")
        with open(output_file, 'rb') as f:
            documents = pickle.load(f)
        return documents

    documents = []
    # Parcourt de tous les fichiers texte du repertoire
    files = [f for f in os.listdir(directory_path) if f.endswith(".txt")]
    total_files = len(files)

    print(f"Prétraitement de {total_files} documents...")

    # Création du dictionnaire doc_id avec tokens prétraités en clé
    for i, filename in enumerate(tqdm(files, desc="Prétraitement")):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()
            tokens = preprocess_text(original_text)
            documents.append({
                "doc_id": filename,
                "tokens": tokens
            })

    # Sauvegarder les résultats dans un fichier pickle
    # Evite de refaire le prétraitement à chaque exécution
    # Notamment lors des futur etapes ou on testera differents modeles d'indexation
    print(f"Sauvegarde des données prétraitées dans {output_file}...")
    with open(output_file, 'wb') as f:
        pickle.dump(documents, f)

    return documents

####################### Main, programme principal pour nettoyer tous les textes ######################
if __name__ == "__main__":
    # Chemin vers le répertoire contenant les fichiers texte
    directory_path = "wiki_split_extract_2k"
    output_file = "preprocessed_data.pkl"

    if not os.path.exists(directory_path):
        print(f"Erreur: Le répertoire '{directory_path}' n'a pas été trouvé.")
    try:
        documents = prepare_data_for_indexing(directory_path, output_file)
        print(f"\nNombre de documents prêts pour l'indexation : {len(documents)}")
        # Exemple d'affichage du premier document
        if documents:
            print("\nExemple de document prétraité :")
            print(f"Doc ID: {documents[0]['doc_id']}")
            print(f"Tokens: {documents[0]['tokens'][:10]}...")  # Afficher les 10 premiers tokens
    except FileNotFoundError:
        print(f"Erreur: Le répertoire '{directory_path}' n'a pas pu être ouvert.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")
