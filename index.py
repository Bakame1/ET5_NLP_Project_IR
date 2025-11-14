def make_reverse_index(documents):
    """Crée un index inversé à partir de la liste des documents prétraités.

    documents: liste de dicts {'doc_id': ..., 'tokens': [...], '_index': int}
    Retour: dict token -> list de doc_id
    """
    reverse_index = {}
    for doc in documents:
        doc_id = doc.get('doc_id')
        tokens = doc.get('tokens', [])
        for token in set(tokens):  # Utiliser set pour éviter les doublons
            reverse_index.setdefault(token, []).append(doc_id)
    return reverse_index


def get_from_reverse_index(reverse_index, token):
    """Récupère la liste des doc_ids contenant le terme donné."""
    return reverse_index.get(token, [])


def get_candidate_docs(reverse_index, documents, request_tokens):
    """Récupère les documents (dicts) correspondant aux termes de la requête.

    Retourne une liste de documents (les éléments de `documents`) dont le `doc_id` apparaît
    dans l'index inversé pour au moins un des `request_tokens`.
    """
    relevant_doc_ids = set()
    for term in request_tokens:
        doc_ids = get_from_reverse_index(reverse_index, term)
        relevant_doc_ids.update(doc_ids)

    # Construire une map doc_id -> document pour retrouver rapidement les dicts
    doc_map = {doc['doc_id']: doc for doc in documents}
    return [doc_map[d] for d in relevant_doc_ids if d in doc_map]
