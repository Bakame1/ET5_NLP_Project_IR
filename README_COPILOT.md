# ET5_NLP_Project_IR — README complet

Ce dépôt implémente un moteur de recherche d'information simple sur des extraits Wikipedia prétraités. Le pipeline combine : prétraitement (spaCy), indexation TF‑IDF (rapide), recherche par similarité cosinus (rapide) et reranking optionnel par Cross‑Encoder (précis). Une évaluation selon le paradigme Cranfield est fournie (P@k, R@k, MRR, MAP).

---

## Vue d'ensemble

Objectif : fournir une chaine complète pour récupérer des documents pertinents à partir de requêtes en français, en combinant rapidité (TF‑IDF + cosine_similarity) et précision (re‑ranking par cross‑encoder). Le pipeline produit aussi des métriques d'évaluation pour juger la qualité de la recherche.

Composants principaux :
- `preprocessing_all_text.py` : prétraitement des textes (nettoyage Wikipedia, tokenisation, lemmatisation, suppression stop-words) et génération de `preprocessed_data.pkl`.
- `tf_idf.py` : construction TF‑IDF, recherche par cosinus, et fonctions de reranking via `sentence-transformers` CrossEncoder.
- `evaluation.py` : implémentation des métriques P@k, R@k, MRR, MAP et d'une fonction `evaluate_all` pour calculer les scores sur l'ensemble des requêtes.
- `pipeline.py` : script tout‑en‑un orchestration : prétraitement, TF‑IDF (ou chargement), recherche, reranking optionnel, puis évaluation et sauvegarde des résultats.
- `requetes.jsonl` : fichier de requêtes + fichiers pertinents (format Cranfield: `Answer file` + `Queries`).
- `wiki_split_extract_2k/` : dossiers de documents bruts (fichiers `.txt`) utilisés pour l'indexation.

Fichiers additionnels générés par le pipeline :
- `preprocessed_data.pkl` : dump des documents prétraités (liste de dicts {"doc_id", "tokens"}).
- `tfidf_model.pkl` : sauvegarde (joblib) de la matrice TF‑IDF et du vectorizer.
- `evaluation_results.json` : résultat de l'évaluation (P@k, R@k, MRR, MAP + détail `per_query`).

---

## Installation et dépendances

Le projet utilise Python 3.8+ (testé avec 3.10+). Les dépendances sont listées dans `requirements.txt`. Le modèle spaCy français (`fr_core_news_sm`) est automatiquement téléchargé par `preprocessing_all_text.py` s'il n'est pas installé.

Exemples de commandes (Windows, `cmd.exe`) :

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Si vous préférez utiliser conda :

```bat
conda create -n et5_ir python=3.10 -y
conda activate et5_ir
pip install -r requirements.txt
```

---

## Utilisation — pipeline principal

Le script central est `pipeline.py`. Il orchestre toutes les étapes et écrit `evaluation_results.json`.

Commandes d'exécution (exemples) :

- Pipeline rapide (sans reranking) :

```bat
python pipeline.py --k 5
```

- Pipeline avec reranking (Cross‑Encoder) — attention téléchargement du modèle et temps d'exécution :

```bat
python pipeline.py --k 5 --rerank --top-k 30
```

Options utiles :
- `--k`: nombre final de résultats retournés par requête (par défaut 10)
- `--top-k`: nombre de documents TF‑IDF à envoyer au cross‑encoder (par défaut 30)
- `--rerank`: active le reranking cross‑encoder
- `--force-preprocess`: force le prétraitement même si `preprocessed_data.pkl` existe
- `--force-tfidf`: force le recalcul TF‑IDF même si `tfidf_model.pkl` existe

Remarques :
- Le reranking nécessite `sentence-transformers` et télécharge un modèle cross‑encoder (ex : `cross-encoder/ms-marco-MiniLM-L-6-v2`). Assurez-vous d'avoir de la bande passante et de la mémoire.
- Le prétraitement génère `preprocessed_data.pkl`. Si vous modifiez `preprocessing_all_text.py`, supprimez ce fichier et relancez avec `--force-preprocess`.

---

## Format des requêtes — `requetes.jsonl`

Chaque ligne contient un JSON avec :
- `Answer file` : nom du fichier `wiki_xxxxxx.txt` considéré comme pertinent pour la/les requête(s)
- `Queries` : liste de variantes textuelles de la requête

Exemple :

```json
{"Answer file": "wiki_066072.txt", "Queries": ["langue roumain", "langue roumanie"]}
```

Le pipeline supporte plusieurs variantes par entrée : il fusionne actuellement les scores TF‑IDF en prenant le meilleur score d'un document parmi les variantes (approche simple). Vous pouvez améliorer la fusion (somme, moyenne, RRF...)

---

## Évaluation (métriques implémentées)

Le module `evaluation.py` calcule :
- Précision@k (P@k) — proportion de documents pertinents dans les k premiers résultats.
- Rappel@k (R@k) — proportion des documents pertinents retrouvés parmi les k premiers résultats.
- Mean Reciprocal Rank (MRR) — moyenne des 1 / rang du premier document pertinent.
- Mean Average Precision (MAP) — moyenne des Average Precision par requête.

Le fichier `evaluation_results.json` contient ces scores globaux et un champ `per_query` avec, pour chaque `qid`, la liste `retrieved` (doc_ids par ordre décroissant) et `relevant`.

---

## Détails d'implémentation importants

- `preprocessing_all_text.py` : utilise spaCy pour tokeniser et lemmatiser, enlève stop-words et tokens courts. Il contient des expressions régulières pour supprimer les « bandeaux » et sections typiques de pages Wikipedia.
- `tf_idf.py` :
  - La fonction `tf_idf(preprocessed_documents)` construit un `TfidfVectorizer` qui attend des listes de tokens (on utilise `tokenizer=identity` et `preprocessor=identity`).
  - `get_top_k_documents` calcule la similarité cosinus entre une requête (prétraitée avec le même pipeline) et la matrice TF‑IDF.
  - `rerank_with_cross_encoder` construit des paires (query, document_text) et utilise `CrossEncoder.predict` pour obtenir des scores fins et trier.
  - `get_top_k_documents_with_reranking` combine les deux étapes (récupérer plus d'items par TF‑IDF, puis reranker et garder les k meilleurs).

---

## Conseils et bonnes pratiques

- Pour le développement itératif, travaillez d'abord sans `--rerank` pour éviter d'attendre le téléchargement et l'inférence du cross‑encoder.
- Gardez `preprocessed_data.pkl` et `tfidf_model.pkl` si vous ne changez pas le pipeline de prétraitement ou la configuration TF‑IDF (gain de temps important).
- Si vous avez peu de mémoire, réduisez `--top-k` lors du reranking.
- Pour évaluer rapidement uniquement la partie TF‑IDF sans exécuter le prétraitement, vous pouvez exécuter `tf_idf.py` en mode `__main__` (existant) ou lancer `test_TF.py` (script de test minimal fourni).

---

## Résolution de problèmes courants

- Erreur JSON lors du parsing de `requetes.jsonl` : le parser du pipeline ignore les préfixes non-JSON (lignes commençant par `//`). Si vous éditez `requetes.jsonl`, conservez des lignes JSON valides par ligne.
- Problèmes spaCy : si le téléchargement automatique échoue, installez manuellement :

```bat
python -m spacy download fr_core_news_sm
```

- Cross‑Encoder trop lent / mémoire insuffisante :
  - Essayez un modèle plus léger (ou désactivez `--rerank`).
  - Réduisez `--top-k`.

---

## Prochaines étapes / améliorations suggérées

- Ajouter une implémentation de Rank Fusion (RRF) pour combiner variantes de requête.
- Ajouter des tests unitaires pour `evaluation.py` (AP, MRR, P@k cases limites).
- Expérimenter avec embeddings (bi-encoder) + FAISS pour accélérer la récupération initiale.
- Générer un rapport comparatif automatique TF‑IDF vs TF‑IDF+Cross‑Encoder.

---

Si vous voulez, je peux :
- Ajouter un petit `README.md` plus court à la racine ou remplacer le `README.md` existant par celui-ci.
- Ajouter des tests unitaires pour `evaluation.py`.
- Lancer une exécution complète avec `--rerank` (prévenir : téléchargement du modèle et temps de calcul). 

Dites-moi quelle option vous préférez pour la suite.
