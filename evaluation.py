from typing import List, Set, Dict
    
"""
Calcule la précision à k (P@k).

@retrieved : liste d'identifiants récupérés ordonnés par score décroissant
@relevant : ensemble d'identifiants pertinents pour la requête
@k : seuil k
@return : précision@k (float)
"""
def precision_at_k(retrieved, relevant, k):
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    relevant_in_k = sum(1 for d in retrieved_k if d in relevant) # Nombre de documents pertinents dans les k premiers
    return relevant_in_k / float(k) # Diviser par k

"""
Calcule le rappel à k (R@k).

@retrieved : liste d'identifiants récupérés ordonnés
@relevant : ensemble d'identifiants pertinents
@k : seuil k
@return : rappel@k (float)
"""
def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_in_k = sum(1 for d in retrieved_k if d in relevant) # Nombre de documents pertinents dans les k premiers
    return relevant_in_k / float(len(relevant)) # Diviser par le nombre total de documents pertinents

"""
Calcule le Reciprocal Rank (RR) pour une requête.

@retrieved : liste d'identifiants récupérés ordonnés
@relevant : ensemble d'identifiants pertinents
@return : RR (float) = 1/position du premier document pertinent, 0 si aucun pertinent
"""
def reciprocal_rank(retrieved, relevant):
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant: # Premier document pertinent trouvé
            return 1.0 / idx
    return 0.0


"""
Calcule l'Average Precision (AP) pour une requête.

@retrieved : liste d'identifiants récupérés ordonnés
@relevant : ensemble d'identifiants pertinents
@return : moyenne des précisions calculées aux positions des documents pertinents
"""
def average_precision(retrieved, relevant):
    if not relevant:
        return 0.0
    num_relevant = 0
    precisions = []
    for idx, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            num_relevant += 1 # Incrémente le nombre de documents pertinents trouvés
            precisions.append(num_relevant / float(idx)) # Précision à cette position
    if not precisions:
        return 0.0
    return sum(precisions) / float(len(relevant)) # On divise par le nombre total de documents pertinents

"""
Calcule la Mean Reciprocal Rank (MRR) sur un ensemble de requêtes.

@list_of_retrieved : liste des listes récupérées par requête
@list_of_relevant : liste des ensembles pertinents par requête (même ordre)
@return : Moyenne des RR sur l'ensemble des requêtes
"""

def mean_reciprocal_rank(list_of_retrieved, list_of_relevant):
    assert len(list_of_retrieved) == len(list_of_relevant) # Vérifier la correspondance des longueurs
    rr_sum = 0.0
    for retrieved, relevant in zip(list_of_retrieved, list_of_relevant):
        rr_sum += reciprocal_rank(retrieved, relevant) # Somme des RR
    return rr_sum / len(list_of_retrieved) if list_of_retrieved else 0.0 # Moyenne des RR


"""
Calcule la Mean Average Precision (MAP) sur un ensemble de requêtes.

@list_of_retrieved : liste des listes récupérées par requête
@list_of_relevant : liste des ensembles pertinents par requête (même ordre)
@return : Moyenne des AP sur l'ensemble des requêtes
"""
def mean_average_precision(list_of_retrieved, list_of_relevant):
    assert len(list_of_retrieved) == len(list_of_relevant) # Vérifier la correspondance des longueurs
    ap_sum = 0.0
    for retrieved, relevant in zip(list_of_retrieved, list_of_relevant):
        ap_sum += average_precision(retrieved, relevant) # Somme des AP
    return ap_sum / len(list_of_retrieved) if list_of_retrieved else 0.0 # Moyenne des AP

"""
Calcule P@k, R@k, MRR et MAP pour un ensemble de requêtes.

@ground_truths : dictionnaire qui relie query_id à set(relevant_doc_ids)
@retrieved_per_query : dictionnaire qui relie query_id à liste ordonnée de doc_ids récupérés
@ks : liste des valeurs de k pour lesquelles calculer P@k et R@k (par défaut [1, 5, 10])
@return : dictionnaire avec les métriques globales
"""
def evaluate_all(ground_truths, retrieved_per_query, ks = None):
    if ks is None: # Valeurs par défaut
        ks = [1, 5, 10]

    list_of_retrieved = list(retrieved_per_query.values()) # Liste des listes récupérées
    list_of_relevant = [ground_truths[qid] for qid in retrieved_per_query.keys()] # Liste des ensembles pertinents

    mean_p_at_k = {k: 0.0 for k in ks}
    mean_r_at_k = {k: 0.0 for k in ks}
    for k in ks:
        mean_p_at_k[k] = sum(precision_at_k(retrieved, relevant, k)
                             for retrieved, relevant in zip(list_of_retrieved, list_of_relevant)) / len(list_of_retrieved) # Moyenne P@k
        mean_r_at_k[k] = sum(recall_at_k(retrieved, relevant, k)
                             for retrieved, relevant in zip(list_of_retrieved, list_of_relevant)) / len(list_of_retrieved) # Moyenne R@k

    mrr = mean_reciprocal_rank(list_of_retrieved, list_of_relevant) # Calcul MRR
    map_score = mean_average_precision(list_of_retrieved, list_of_relevant) # Calcul MAP

    return {
        "P@k": mean_p_at_k,
        "R@k": mean_r_at_k,
        "MRR": mrr,
        "MAP": map_score
    }
