import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import preprocess
import joblib
from sentence_transformers import CrossEncoder
from functools import lru_cache

# Fonction identité pour le vectoriseur TF-IDF
def identity(x):
    return x

def tf_idf(preprocessed_documents):
    """
    Calcule la matrice TF-IDF pour les documents prétraités.
    @preprocessed_documents : liste de documents prétraités (chaque document est un dict avec 'doc_id' et 'tokens')
    @return : matrice TF-IDF et le vectoriseur utilisé
    """

    # Préparer les données pour TF-IDF
    #corpus =[" ".join(doc['tokens']) for doc in preprocessed_documents]
    corpus = [doc["tokens"] for doc in preprocessed_documents]
    '''
    Initialiser le vectoriseur TF-IDF
    1. tokenizer=lambda x: x : utilise les tokens déjà prétraités (liste de tokens)
    2. preprocessor=lambda x: x : pas de prétraitement supplémentaire
    3. lowercase=False : pas de mise en minuscule
    @max_df=0.4 : ignore les termes présents dans plus de 40% des documents
    '''
    vectorizer = TfidfVectorizer(
        tokenizer=identity,
        preprocessor=identity,
        lowercase=False,
        token_pattern=None,
        max_df=0.4,
    )
    # on applique TF-IDF sur le corpus
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

'''
Retourne les k documents les plus similaires à la requête

@X : la matrice TF-IDF des documents
@vectorizer : le vectoriseur TF-IDF
@query : la requête utilisateur
@documents : la liste des documents prétraités
@k : le nombre de documents à retourner
@return : une liste de tuples (doc_id, similarité)
'''
def get_top_k_documents(X,vectorizer,query,documents, k):
    # Prétraiter la requête avec le même pipeline que les documents
    query_pre_processed = preprocess.preprocess_text(query)
    # Convertir la requête prétraitée en vecteur TF-IDF
    query_vec = vectorizer.transform([query_pre_processed])
    # Calculer la similarité cosinus entre la requête et tous les documents
    similarities = cosine_similarity(query_vec, X).flatten()
    top_k_indices = similarities.argsort()[-k:][::-1]
    top_k_documents = [(documents[i]["doc_id"], similarities[i]) for i in top_k_indices]
    return top_k_documents


"""
Re-classe les documents issus du TF-IDF en utilisant un cross-encoder pour une meilleure précision.

@query : la requête utilisateur
@top_k_documents : liste de tuples (doc_id, similarité_tfidf) issus du TF-IDF
@documents : la liste des documents prétraités
@model_name : nom du modèle cross-encoder à utiliser
@return : une liste de tuples (doc_id, score_cross_encoder) triée par score décroissant
"""
def rerank_with_cross_encoder(query, top_k_documents, documents, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    cross_encoder = get_cross_encoder(model_name)

    doc_dict = {doc["doc_id"]: doc["tokens"] for doc in documents} # Créer un dictionnaire pour accéder rapidement aux tokens par doc_id

    # Préparer les paires (query, document_text) que le cross-encoder attend en entrée
    pairs = []
    doc_ids = []
    for doc_id, _ in top_k_documents:
        doc_text = " ".join(doc_dict[doc_id]) # Reconstruire le texte à partir des tokens pour le cross-encoder
        pairs.append([query, doc_text])
        doc_ids.append(doc_id)

    scores = cross_encoder.predict(pairs) # Calculer les scores avec le cross-encoder
    # score de cross-encoder : un score entre 0 et 1 indiquant la pertinence du document par rapport à la requête

    reranked_documents = [(doc_ids[i], float(scores[i])) for i in range(len(doc_ids))] # Créer une liste de tuples (doc_id, score)
    reranked_documents.sort(key=lambda x: x[1], reverse=True) # Trier par score décroissant

    return reranked_documents


@lru_cache(maxsize=2)
def get_cross_encoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
    """Charge et met en cache un cross-encoder."""
    return CrossEncoder(model_name)


def save_cross_encoder(model, output_dir='cross_encoder_model'):
    """Sauvegarde le modèle cross-encoder dans le répertoire spécifié."""
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)


def load_cross_encoder(output_dir='cross_encoder_model'):
    """Charge le modèle cross-encoder depuis le répertoire spécifié."""
    if os.path.exists(output_dir):
        return CrossEncoder(output_dir)
    raise FileNotFoundError(f"Cross-encoder non trouvé dans {output_dir}")


"""
Fonction complète combinant TF-IDF (rapide) et cross-encoder (précis).
1. Récupère les top k meilleurs documents via TF-IDF + cosine similarity (rapide) pour ensuite les re-classer
2. Re-classe ces documents avec un cross-encoder (précis)
3. Retourne les k meilleurs documents finaux

@X : la matrice TF-IDF des documents
@vectorizer : le vectoriseur TF-IDF
@query : la requête utilisateur
@documents : la liste des documents prétraités
@k : le nombre final de documents à retourner
@top_k_for_reranking : nombre de documents à récupérer avant reranking (par défaut: k*3)
@return : une liste de tuples (doc_id, score_cross_encoder)
"""
def get_top_k_documents_with_reranking(X, vectorizer, query, documents, k, top_k_for_reranking=None):
    if top_k_for_reranking is None: # Par défaut, on récupère 3 fois plus de documents que souhaité pour le reranking
        top_k_for_reranking = min(k * 3, len(documents))

    # Étape 1: Récupération des top documents via TF-IDF (rapide)
    print(f"Étape 1: Récupération de {top_k_for_reranking} documents via TF-IDF...")
    top_k_tfidf = get_top_k_documents(X, vectorizer, query, documents, top_k_for_reranking)

    # Étape 2: Re-classement avec cross-encoder (précis)
    print(f"Étape 2: Re-classement avec cross-encoder...")
    reranked = rerank_with_cross_encoder(query, top_k_tfidf, documents)

    # Étape 3: Retourner les k meilleurs
    return reranked[:k]

def save_tfidf_model(X, vectorizer, model_file="tfidf_model.pkl"):
    """
    Sauvegarde le modèle TF-IDF dans un fichier pickle.
    """
    with open(model_file, 'wb') as f:
        joblib.dump({'X':X,'vectorizer':vectorizer}, f)

def load_tfidf_model(model_file="tfidf_model.pkl"):
    """
    Charge le modèle TF-IDF depuis un fichier pickle.
    """
    with open(model_file, 'rb') as f:
        data = joblib.load(model_file)
        return data["X"], data["vectorizer"]

if __name__ == "__main__":
    # exemple d'utilisation
    # Charger les documents prétraités (créé par preprocessing_all_text)
    if not os.path.exists('preprocessed_data.pkl'):
        raise SystemExit("Fichier 'preprocessed_data.pkl' introuvable. Lancez d'abord le prétraitement ou générez ce fichier.")
    with open('preprocessed_data.pkl', 'rb') as f:
        documents = pickle.load(f)

    # Charger ou calculer TF-IDF de manière robuste
    if os.path.isfile("tfidf_model.pkl"):
        print("Chargement du modèle TF-IDF depuis tfidf_model.pkl...")
        try:
            X, vectorizer = load_tfidf_model('tfidf_model.pkl')
        except Exception:
            print("Echec du chargement du modèle TF-IDF (format inattendu). Recalcul en cours...")
            X, vectorizer = tf_idf(documents)
            save_tfidf_model(X, vectorizer)
    else:
        X, vectorizer = tf_idf(documents)
        save_tfidf_model(X, vectorizer)

    query = "qu'est-ce que la 6e armée"

    # Approche 1: TF-IDF + Cosine Similarity seul (rapide)
    print(f"\n{'='*70}")
    print(f"Approche 1: TF-IDF + Cosine Similarity (RAPIDE)")
    print(f"{'='*70}")
    top_k = get_top_k_documents(X, vectorizer, query, documents, k=5)
    print(f"Top {len(top_k)} documents pour la requête '{query}':")
    for doc_id, score in top_k:
        print(f"  Doc ID: {doc_id}, Similarité TF-IDF: {score:.4f}")

    # Approche 2: TF-IDF + Cosine Similarity (rapide) puis Cross-Encoder (précis)
    print(f"\n{'='*70}")
    print(f"Approche 2: TF-IDF (RAPIDE) + Cross-Encoder (PRÉCIS)")
    print(f"{'='*70}")
    top_k_reranked = get_top_k_documents_with_reranking(
        X, vectorizer, query, documents, k=5, top_k_for_reranking=15
    )
    print(f"Top {len(top_k_reranked)} documents après reranking pour la requête '{query}':")
    for doc_id, score in top_k_reranked:
        print(f"  Doc ID: {doc_id}, Score Cross-Encoder: {score:.4f}")
