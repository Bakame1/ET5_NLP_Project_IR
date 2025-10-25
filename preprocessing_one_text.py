import spacy
import re
import string
import os

#Configuration des stop words et du modèle spaCy
#Charger le modèle français de spaCy si non présent, le télécharger
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

############################# Stop words francais #############################
# Obtenir les stop words français
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

####################### Main, programme principal pour nettoyer un texte ######################
if __name__ == "__main__":
    # Chemin vers le fichier texte
    example_file_path = "wiki_split_extract_2k/wiki_000711.txt"
    if not os.path.exists(example_file_path):
        print(f"Erreur: Le fichier d'exemple '{example_file_path}' n'a pas été trouvé.")
    try:
        # Lecture du fichier texte
        with open(example_file_path, 'r', encoding='utf-8') as f:
            original_text = f.read()

        # Afficher texte original
        print(f"--- Texte Original ({example_file_path}) ---")
        print(original_text)
        print("\n" + "="*30 + "\n")

        # Afficher texte après suppression du bruit Wiki
        print("--- Texte après suppression du bruit Wiki ---")
        text_after_noise_removal = remove_wiki_noise(original_text)
        print(text_after_noise_removal)
        print("\n" + "="*30 + "\n")

        # Appliquer le prétraitement complet
        processed_tokens = preprocess_text(original_text)
        print("--- Tokens Prétraités (Lemmes) ---")
        print(processed_tokens)
        print(f"\nNombre de tokens après prétraitement : {len(processed_tokens)}")

    except FileNotFoundError:
        print(f"Erreur: Le fichier '{example_file_path}' n'a pas pu être ouvert.")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")