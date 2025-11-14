# Projet IR - Recherche d'Information avec TF-IDF et Cross-Encoder

Version Python 3.8 ou plus

## Installation

### Activer le venv 
```bash
.venv\Scripts\activate
```

### Installer les dépendances
```bash
pip install -r requirements.txt
```

### Télécharger le modèle spaCy français
```bash
python -m spacy download fr_core_news_sm
```

## Utilisation

### 1. Interface Streamlit (Recommandé)

L'interface Streamlit offre deux modes d'utilisation :

#### Lancer l'interface
```bash
streamlit run app.py
```

#### Mode Recherche Interactive
- Entrez une requête dans la barre de recherche
- Configurez les paramètres (k, top-k, reranking) dans la barre latérale
- Cliquez sur "Rechercher" pour obtenir les documents pertinents
- Visualisez les résultats avec scores et extraits de documents

#### Mode Évaluation Complète
- Configure les paramètres dans la barre latérale
- Lancez l'évaluation sur toutes les requêtes du fichier `requetes.jsonl`
- Consultez les métriques (MRR, MAP, P@k, R@k)
- Téléchargez les résultats en JSON

### 2. Pipeline en ligne de commande

```bash
python pipeline.py --k 10 --top-k 30 --rerank
```

**Options disponibles :**
- `--k` : Nombre de résultats finaux par requête (défaut: 10)
- `--top-k` : Nombre de documents TF-IDF à envoyer au cross-encoder (défaut: 30)
- `--rerank` : Activer le reranking avec cross-encoder
- `--force-preprocess` : Forcer le prétraitement des documents
- `--force-tfidf` : Forcer le recalcul du modèle TF-IDF

### 3. Modules individuels

#### Prétraitement seul
```bash
python preprocessing_all_text.py
```

#### Indexation et recherche
```bash
python indexation.py
```

## Architecture du Projet

### Fichiers principaux

- **`app.py`** : Interface Streamlit avec recherche interactive et évaluation
- **`pipeline.py`** : Pipeline complet (prétraitement → TF-IDF → reranking → évaluation)
- **`preprocessing_all_text.py`** : Prétraitement des documents Wikipedia
- **`indexation.py`** : TF-IDF, similarité cosinus et cross-encoder
- **`evaluation.py`** : Métriques d'évaluation (P@k, R@k, MRR, MAP)

### Fichiers générés

- **`preprocessed_data.pkl`** : Documents nettoyés et tokenisés (réutilisable)
- **`tfidf_model.pkl`** : Modèle TF-IDF sauvegardé
- **`evaluation_results.json`** : Résultats des métriques d'évaluation

### Données

- **`wiki_split_extract_2k/`** : Corpus de documents Wikipedia
- **`requetes.jsonl`** : Requêtes de test avec documents pertinents (ground truth)

## Workflow du Pipeline

1. **Prétraitement** : Nettoyage, tokenisation, lemmatisation avec spaCy
2. **Indexation TF-IDF** : Vectorisation des documents
3. **Recherche rapide** : Similarité cosinus pour sélectionner top-k documents
4. **Reranking (optionnel)** : Cross-encoder pour améliorer la précision
5. **Évaluation** : Calcul des métriques P@k, R@k, MRR, MAP

## Métriques d'Évaluation

- **P@k** : Précision aux k premiers résultats
- **R@k** : Rappel aux k premiers résultats  
- **MRR** : Mean Reciprocal Rank (position du 1er document pertinent)
- **MAP** : Mean Average Precision

## Notes sur le Prétraitement

### Fichiers à tester avec des phrases inutiles de Wikipedia
- wiki_000711
- wiki_000297
- wiki_000468
- wiki_000612

Le prétraitement supprime automatiquement les bandeaux Wikipedia et autres contenus non pertinents.

## Performance

- **TF-IDF seul** : Très rapide (~ms par requête)
- **TF-IDF + Cross-Encoder** : Plus précis mais plus lent (~secondes par requête)

**Recommandation** : Utiliser `top-k` = 3-5× le nombre de résultats finaux pour un bon compromis vitesse/précision.
