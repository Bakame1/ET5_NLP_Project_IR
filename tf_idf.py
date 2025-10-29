import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import preprocessing_all_text as preprocess
import joblib

def identity(x):
    return x

def tf_idf(preprocessed_documents):

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
    #exemple d'utilisation
    documents = pickle.load(open('preprocessed_data.pkl', 'rb'))
    if (os.path.isfile("tfidf_model.pkl")):
        print("Chargement du modèle TF-IDF depuis tfidf_model.pkl...")
        with open("tfidf_model.pkl", 'rb') as f:
            X, vectorizer = pickle.load(f)
    else:
        X, vectorizer = tf_idf(documents)
        save_tfidf_model(X, vectorizer)
    query="qu'est-ce que la 6e armée"
    top_k = get_top_k_documents(X, vectorizer, query, documents, k=5)
    print(f"Top {len(top_k)} documents pour la requête '{query}':")
    for doc_id, score in top_k:
        print(f"Doc ID: {doc_id}, Similarité: {score:.4f}")