# Cross-Encoder pour améliorer la recherche documentaire

## Vue d'ensemble

Ce projet combine maintenant deux approches pour une recherche documentaire optimale :

1. **TF-IDF + Cosine Similarity** : Approche rapide pour filtrer les documents pertinents
2. **Cross-Encoder** : Approche précise pour re-classer les meilleurs candidats

## Architecture

```
Requête utilisateur
    ↓
[TF-IDF + Cosine Similarity] ← Étape 1 : RAPIDE
    ↓
Top N documents (ex: 15 documents)
    ↓
[Cross-Encoder] ← Étape 2 : PRÉCIS
    ↓
Top K documents finaux (ex: 5 documents)
```

## Installation

Installez d'abord les dépendances nécessaires :

```bash
pip install sentence-transformers
```

Ou installez toutes les dépendances du projet :

```bash
pip install -r requirements.txt
```

## Utilisation

### Méthode 1 : TF-IDF seul (rapide)

```python
import pickle
from indexation import tf_idf, get_top_k_documents

# Charger les documents
documents = pickle.load(open('preprocessed_data.pkl', 'rb'))

# Créer le modèle TF-IDF
X, vectorizer = tf_idf(documents)

# Rechercher les documents
query = "qu'est-ce que la 6e armée"
top_k = get_top_k_documents(X, vectorizer, query, documents, k=5)

for doc_id, score in top_k:
    print(f"Doc ID: {doc_id}, Score: {score:.4f}")
```

### Méthode 2 : TF-IDF + Cross-Encoder (précis)

```python
import pickle
from indexation import tf_idf, get_top_k_documents_with_reranking

# Charger les documents
documents = pickle.load(open('preprocessed_data.pkl', 'rb'))

# Créer le modèle TF-IDF
X, vectorizer = tf_idf(documents)

# Rechercher avec reranking
query = "qu'est-ce que la 6e armée"
top_k = get_top_k_documents_with_reranking(
    X, vectorizer, query, documents,
    k=5,  # Nombre de résultats finaux
    top_k_for_reranking=15  # Nombre de candidats pour le reranking
)

for doc_id, score in top_k:
    print(f"Doc ID: {doc_id}, Score: {score:.4f}")
```

### Méthode 3 : Reranking manuel

```python
import pickle
from indexation import tf_idf, get_top_k_documents, rerank_with_cross_encoder

# Charger les documents
documents = pickle.load(open('preprocessed_data.pkl', 'rb'))

# Créer le modèle TF-IDF
X, vectorizer = tf_idf(documents)

# Étape 1 : TF-IDF
query = "qu'est-ce que la 6e armée"
candidates = get_top_k_documents(X, vectorizer, query, documents, k=20)

# Étape 2 : Cross-Encoder
reranked = rerank_with_cross_encoder(query, candidates, documents)

# Prendre les 5 meilleurs
for doc_id, score in reranked[:5]:
    print(f"Doc ID: {doc_id}, Score: {score:.4f}")
```

## Fonctions disponibles

### `get_top_k_documents(X, vectorizer, query, documents, k)`
Recherche rapide par TF-IDF + similarité cosinus.
- **Avantage** : Très rapide
- **Inconvénient** : Moins précis sémantiquement

### `rerank_with_cross_encoder(query, top_k_documents, documents, model_name)`
Re-classe des documents avec un cross-encoder.
- **Avantage** : Très précis sémantiquement
- **Inconvénient** : Plus lent (calcul intensif)
- **Modèle par défaut** : `cross-encoder/ms-marco-MiniLM-L-6-v2`

### `get_top_k_documents_with_reranking(X, vectorizer, query, documents, k, top_k_for_reranking)`
Combine les deux approches automatiquement.
- **k** : Nombre de documents finaux à retourner
- **top_k_for_reranking** : Nombre de candidats à récupérer avant reranking (par défaut : k*3)

## Paramètres recommandés

Pour de bonnes performances :
- `k=5` : 5 résultats finaux
- `top_k_for_reranking=15` : 15 candidats pour le reranking (3x plus)

Pour une précision maximale :
- `k=10` : 10 résultats finaux
- `top_k_for_reranking=30` : 30 candidats pour le reranking

## Modèles Cross-Encoder disponibles

Vous pouvez changer le modèle cross-encoder :

```python
# Modèle par défaut (recommandé pour français/anglais)
rerank_with_cross_encoder(query, candidates, documents, 
    model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

# Modèle plus grand et plus précis (mais plus lent)
rerank_with_cross_encoder(query, candidates, documents, 
    model_name='cross-encoder/ms-marco-MiniLM-L-12-v2')

# Modèle multilingual
rerank_with_cross_encoder(query, candidates, documents, 
    model_name='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
```

## Performances

Exemple de comparaison sur une requête :

**TF-IDF seul** :
- Temps : ~0.1s
- Précision : Bonne

**TF-IDF + Cross-Encoder (15 candidats → 5 résultats)** :
- Temps : ~0.5s
- Précision : Excellente

Le cross-encoder analyse la relation sémantique entre la requête et le document, ce qui améliore significativement la qualité des résultats pour les requêtes complexes.

