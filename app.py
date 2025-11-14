import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd

import pipeline
import ranking

# Constantes
DATA_DIR = Path('wiki_split_extract_2k')
PICKLE_PATH = Path('preprocessed_data.pkl')
TFIDF_PATH = Path('tfidf_model.pkl')
RESULTS_PATH = Path('evaluation_results.json')
QUERIES_PATH = Path('requetes.jsonl')
DEFAULT_KS = [1, 5, 10]


@st.cache_resource(show_spinner=False)
def get_cached_documents(force_preprocess):
    """Charge les documents sans appel à st.toast."""
    return pipeline.load_or_preprocess_documents(
        directory_path=str(DATA_DIR),
        output_file=str(PICKLE_PATH),
        force_preprocess=force_preprocess,
        log_fn=lambda msg: None,  # Ne rien faire pour éviter les problèmes de cache
    )


@st.cache_resource(show_spinner=False)
def get_cached_tfidf(_documents, force_tfidf):
    """Charge le TF-IDF sans appel à st.toast."""
    return pipeline.load_or_compute_tfidf(
        _documents,
        model_file=str(TFIDF_PATH),
        force_tfidf=force_tfidf,
        log_fn=lambda msg: None,  # Ne rien faire pour éviter les problèmes de cache
    )


def run_full_pipeline(k, top_k, rerank, force_preprocess, force_tfidf):
    """Exécute le pipeline streamlit"""
    
    class Args:
        pass

    args = Args()
    args.k = k
    args.top_k = top_k
    args.rerank = rerank
    args.force_preprocess = force_preprocess
    args.force_tfidf = force_tfidf

    logs = []

    def log_fn(msg):
        """Fonction de log pour capturer les messages."""
        logs.append(msg)

    # On execute le pipeline principal
    output = pipeline.run_pipeline(args, log_fn=log_fn)

    return output, logs


def search_query(query, documents, X, vectorizer, k, rerank, top_k_for_reranking):
    """Effectue une recherche pour une requête unique.
    @query : texte de la requête
    @documents : liste des documents prétraités
    @X : matrice TF-IDF des documents
    @vectorizer : le vectoriseur TF-IDF
    @k : nombre de documents à retourner
    @rerank : booléen, si True on utilise le cross-encoder pour reranker
    @top_k_for_reranking : nombre de documents à récupérer avant reranking
    @return : liste de tuples (doc_id, score_cross_encoder)
    """
    # S'assurer que chaque document a un '_index' pour le reranking
    for idx, doc in enumerate(documents):
        if isinstance(doc, dict) and not isinstance(doc.get('_index'), int):
            doc['_index'] = idx

    try:
        if rerank:
            top_tfidf = ranking.get_top_k_documents(X, vectorizer, query, documents, top_k_for_reranking)
            results = ranking.rerank_with_cross_encoder(query, top_tfidf, documents)
            return results[:k]
        else:
            return ranking.get_top_k_documents(X, vectorizer, query, documents, k)
    except Exception as e:
        # En cas d'erreur, afficher un message dans Streamlit
        try:
            import streamlit as _st
            _st.error(f"Erreur recherche: {e}")
        except Exception:
            pass
        return []

# Programme principal Streamlit
def main():
    # Titre
    st.set_page_config(page_title='IR Pipeline Streamlit', layout='wide')
    st.title('Moteur de Recherche Wikipedia')

    # Sidebar pour selection les valeurs des paramètres
    with st.sidebar:
        st.header('⚙️ Configuration')

        st.subheader('Paramètres de recherche')

        k = st.slider('Nombre de résultats finaux (k)', min_value=1, max_value=50, value=10)  # nombre de résultats finaux k
        top_k = st.slider('Top-k pour reranking', min_value=k, max_value=150, value=max(30, k))# nombre de documents à récupérer avant reranking
        rerank = st.checkbox('Activer le reranking cross-encoder', value=False)# box pour activer/désactiver le reranking

        st.subheader('Options avancées')
        force_preprocess = st.checkbox('Forcer le prétraitement', value=False)# box pour forcer le prétraitement des documents
        force_tfidf = st.checkbox('Forcer le recalcul TF-IDF', value=False)# box pour forcer le recalcul du modèle TF-IDF

        st.divider()
        st.info('💡 Configurez les options puis utilisez les onglets ci-dessus.')

    # tab1 : Recherche interactive comme sur un moteur de recherche
    # tab2 : Évaluation complète du pipeline sur toutes les requêtes du fichier JSONL
    tab1, tab2 = st.tabs(['🔎 Recherche Interactive', '📊 Évaluation Complète'])

    ################### Onglet 1 : Recherche Interactive ###################
    with tab1:
        st.header('Recherche de documents')
        st.write('Entrez une requête pour rechercher des documents pertinents dans la collection.')

        # Champ de saisie de la requête
        query = st.text_input('Votre requête:', placeholder='Ex: qu\'est-ce que la 6e armée')

        col1, col2 = st.columns([1, 4])
        with col1:
            # Bouton de recherche
            search_button = st.button('🚀 Rechercher', type='primary', use_container_width=True)

        # Exécution de la recherche
        if search_button and query:
            with st.spinner('Chargement des données...'):
                documents = get_cached_documents(force_preprocess)
                X, vectorizer = get_cached_tfidf(documents, force_tfidf)

            with st.spinner('Recherche en cours...'):
                start_time = time.time()
                results = search_query(query, documents, X, vectorizer, k, rerank, top_k)
                search_time = time.time() - start_time

            st.success(f'✅ Recherche terminée en {search_time:.2f}s')

            st.subheader(f'Top {len(results)} résultats pour: "{query}"')

            # Affichage des résultats
            if results:
                df_results = pd.DataFrame([
                    {
                        'Rang': i + 1,
                        'Document ID': doc_id,
                        'Score': f'{score:.4f}'
                    }
                    for i, (doc_id, score) in enumerate(results)
                ])

                st.dataframe(df_results, use_container_width=True, hide_index=True)

                # Visualisation des détails des documents
                st.subheader('📄 Détails des documents')
                for i, (doc_id, score) in enumerate(results[:5]):
                    with st.expander(f'#{i+1} - {doc_id} (Score: {score:.4f})'):
                        doc = next((d for d in documents if d['doc_id'] == doc_id), None)
                        if doc:
                            tokens = doc['tokens']
                            st.write(f'**Nombre de tokens:** {len(tokens)}')
                            st.write(f'**Extrait (100 premiers tokens):**')
                            st.text(' '.join(tokens[:100]) + '...')
                        else:
                            st.warning('Document non trouvé dans le corpus.')
            else:
                st.warning('Aucun résultat trouvé.')

        elif search_button and not query:
            st.warning('⚠️ Veuillez entrer une requête.')

    ################### Onglet 2 : Évaluation sur le fichier JSONL ###################
    with tab2:
        st.header('Évaluation du pipeline complet')
        st.write('Lancez l\'évaluation sur toutes les requêtes du fichier `requetes.jsonl`.')

        # Bouton pour lancer l'évaluation
        run_button = st.button('Lancer le pipeline d\'évaluation', type='primary')

        # Exécution de l'évaluation
        if run_button:
            with st.spinner('Exécution du pipeline...'):
                start = time.time()
                output, logs = run_full_pipeline(k, top_k, rerank, force_preprocess, force_tfidf)
                duration = time.time() - start

            st.success(f'✅ Pipeline terminé en {duration:.1f}s')

            # Affichage des résultats
            with st.expander('📋 Journal d\'exécution'):
                st.code('\n'.join(logs) or 'Aucun log')

            st.subheader('📊 Métriques d\'évaluation')

            metrics = output['metrics']

            col1, col2, col3, col4 = st.columns(4)

            # Affichage des métriques principales
            with col1:
                st.metric('MRR', f"{metrics['MRR']:.4f}")
            with col2:
                st.metric('MAP', f"{metrics['MAP']:.4f}")
            with col3:
                st.metric('P@1', f"{metrics['P@k'][1]:.4f}")
            with col4:
                st.metric('P@5', f"{metrics['P@k'][5]:.4f}")

            st.subheader('Précision et Rappel par k')
            metrics_df = pd.DataFrame({
                'k': list(metrics['P@k'].keys()),
                'Précision@k': [f"{v:.4f}" for v in metrics['P@k'].values()],
                'Rappel@k': [f"{v:.4f}" for v in metrics['R@k'].values()]
            })
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            with st.expander('📑 Résultats détaillés par requête'):
                for qid, docs in output['retrieved'].items():
                    st.write(f'**{qid}**')
                    st.write(', '.join(docs[:10]))

            with st.expander('📄 Métriques complètes (JSON)'):
                st.json(metrics)

            st.download_button(
                label='⬇️ Télécharger les résultats JSON',
                data=json.dumps(metrics, ensure_ascii=False, indent=2),
                file_name='evaluation_results.json',
                mime='application/json'
            )


if __name__ == '__main__':
    main()
