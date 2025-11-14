import os
import json
import argparse
import pickle
from typing import Dict, List, Set, Tuple
from typing import Callable

import preprocess
import indexation
import evaluation


"""
    Charge `requetes.jsonl` et retourne un mapping query_id -> set(relevant_doc_ids).
    On utilisera comme query_id la première 'Query' texte pour l'identification.
    
    @file_path: chemin vers le fichier JSONL des requêtes
    @return: (ground_truths, query_texts) où
        ground_truths: Dict[query_id, Set[relevant_doc_ids]]
        query_texts: Dict[query_id, List[query_variants]]
"""
def load_queries(file_path):
    ground_truths = {} # mapping query_id -> set of relevant doc_ids (ground truth)
    query_texts = {}  # mapping query_id -> list of query variants (pour recherche)
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            # Certaines lignes peuvent contenir un préfixe de commentaire (ex: "// filepath: ...")
            # On récupère la première accolade ouvrante pour isoler l'objet JSON
            start = line.find('{')
            if start == -1:
                continue
            json_part = line[start:]
            try:
                obj = json.loads(json_part)
            except json.JSONDecodeError:
                # Ignore les lignes malformées
                continue
            answer = obj.get("Answer file")
            queries = obj.get("Queries", [])
            # Choisir un qid unique, ici on prend le numéro de ligne
            qid = f"q{line_num}"
            # Stocker le ground truth (set) — parfois il peut y en avoir plusieurs
            ground_truths[qid] = {answer} if answer else set()
            # Stocker les variantes de la requête
            query_texts[qid] = queries
    return ground_truths, query_texts

"""
    Pour une requête (avec variantes), retourne la liste ordonnée de doc_ids.
    On combine les scores pour plusieurs variantes en prenant l'ordre moyen: ici simple approche = prendre la meilleure réponse parmi variantes.
    
    @X: matrice TF-IDF des documents
    @vectorizer: l'objet vectorizer TF-IDF
    @documents: liste des documents (pour mapping id -> texte)
    @query_variants: liste des variantes de la requête
    @k: nombre de documents finaux à retourner
    @rerank: booléen indiquant si on doit reranker avec cross-encoder
    @top_k_for_reranking: nombre de documents à envoyer au cross-encoder pour reranking
    @return: liste ordonnée de doc_ids
"""
def single_query_retrieve(X, vectorizer, documents, query_variants, k, rerank, top_k_for_reranking):
    # On va récupérer des résultats pour chaque variante puis fusionner
    aggregated_scores = {}  # doc_id -> best_score (or sum)
    for q in query_variants:
        top = indexation.get_top_k_documents(X, vectorizer, q, documents, k=len(documents)) # top contient tuples (doc_id, score)
        for doc_id, score in top:
            if doc_id not in aggregated_scores or score > aggregated_scores[doc_id]:
                aggregated_scores[doc_id] = score
    sorted_by_score = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True) # Trier par score tf-idf décroissant
    retrieved_ids = [doc_id for doc_id, _ in sorted_by_score]

    if rerank:
        top_for_rerank = [(doc_id, aggregated_scores[doc_id]) for doc_id in retrieved_ids[:top_k_for_reranking]] # Préparer les top documents pour reranking
        reranked = indexation.rerank_with_cross_encoder(query_variants[0], top_for_rerank, documents) # Utiliser la première variante pour le reranking
        final_ids = [doc_id for doc_id, _ in reranked] # Extraire les doc_ids rerankés
    else: # Pas de reranking, on garde l'ordre TF-IDF
        final_ids = retrieved_ids

    return final_ids[:k]


def load_or_preprocess_documents(directory_path = 'wiki_split_extract_2k', output_file = 'preprocessed_data.pkl', force_preprocess = False, log_fn = print):
    """Charge les documents prétraités depuis pickle ou relance le prétraitement."""
    if os.path.exists(output_file) and not force_preprocess:
        log_fn(f"Chargement des données prétraitées depuis {output_file}...")
        with open(output_file, 'rb') as f:
            return pickle.load(f)
    log_fn("Lancement du prétraitement des documents...")
    return preprocess.prepare_data_for_indexing(directory_path, output_file)


def load_or_compute_tfidf(documents, model_file = 'tfidf_model.pkl', force_tfidf = False, log_fn = print):
    """Charge le modèle TF-IDF ou le recalcule si nécessaire."""
    if os.path.exists(model_file) and not force_tfidf:
        log_fn(f"Chargement du modèle TF-IDF depuis {model_file}...")
        return indexation.load_tfidf_model(model_file)
    log_fn("Calcul du TF-IDF...")
    X, vectorizer = indexation.tf_idf(documents)
    indexation.save_tfidf_model(X, vectorizer, model_file=model_file)
    return X, vectorizer


def retrieve_all_queries(X, vectorizer, documents, query_texts, k, rerank, top_k_for_reranking):
    """Exécute la récupération (et reranking optionnel) pour toutes les requêtes."""
    retrieved_per_query = {}
    for qid, variants in query_texts.items():
        retrieved_per_query[qid] = single_query_retrieve(
            X,
            vectorizer,
            documents,
            variants,
            k=k,
            rerank=rerank,
            top_k_for_reranking=top_k_for_reranking,
        )
    return retrieved_per_query

# Pipeline principal executant toutes les étapes du projet
def run_pipeline(args, log_fn = print):
    # 1) Prétraitement si nécessaire
    documents = load_or_preprocess_documents(
        directory_path='wiki_split_extract_2k',
        output_file='preprocessed_data.pkl',
        force_preprocess=args.force_preprocess,
        log_fn=log_fn,
    )

    # 2) Charger ou calculer TF-IDF
    X, vectorizer = load_or_compute_tfidf(
        documents,
        model_file='tfidf_model.pkl',
        force_tfidf=args.force_tfidf,
        log_fn=log_fn,
    )

    # 3) Charger les queries et ground-truths
    ground_truths, query_texts = load_queries('requetes.jsonl')

    # 4) Pour chaque requête, récupérer et (optionnel) reranker
    retrieved_per_query = retrieve_all_queries(
        X,
        vectorizer,
        documents,
        query_texts,
        k=args.k,
        rerank=args.rerank,
        top_k_for_reranking=args.top_k,
    )

    # 5) Calcul des métriques
    results = evaluation.evaluate_all(ground_truths, retrieved_per_query, ks=[1, 5, 10])

    # 6) Sauvegarder résultats
    log_fn("Résultats :")
    log_fn(json.dumps(results, indent=2, ensure_ascii=False))
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "metrics": results,
        "retrieved": retrieved_per_query,
        "ground_truths": ground_truths,
        "query_texts": query_texts,
        "documents": documents,
    }

# Point d'entrée principal via le terminal
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline recherche: preprocessing -> tf-idf -> (rerank) -> évaluation')
    parser.add_argument('--k', type=int, default=10, help='Nombre de résultats finaux par requête')
    parser.add_argument('--top-k', type=int, default=30, help='Nombre de documents TF-IDF à envoyer au cross-encoder')
    parser.add_argument('--rerank', action='store_true', help='Activer le reranking cross-encoder')
    parser.add_argument('--force-preprocess', action='store_true', help='Forcer le prétraitement même si pickle existe')
    parser.add_argument('--force-tfidf', action='store_true', help='Forcer le recalcul TF-IDF')
    args = parser.parse_args()
    run_pipeline(args)
