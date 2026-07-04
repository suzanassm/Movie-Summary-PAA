import numpy as np
import pandas as pd
import pickle
import re
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_response import llm  # Importar o LLM para reescrita da query
import hnswlib

loaded = False

def load_models():
    global loaded, df, w2v_model, bigram_phraser, sbert_model, tfidf, tfidf_matrix
    global w2v_embeddings, index

    if loaded:
        return

    # Carrega dados
    df = pd.read_csv('data/processed.csv')

    # Carrega modelos
    w2v_model = Word2Vec.load('models/w2v_model.bin')
    bigram_phraser = Phraser.load('models/bigram_phraser.pkl')
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    with open('models/tfidf.pkl', 'rb') as f:
        tfidf, tfidf_matrix = pickle.load(f)

    #w2v_embeddings = np.load('models/w2v_embeddings.npy')
    w2v_embeddings = np.load('models/w2v_embeddings.npy', mmap_mode='r')

    # Carrega índice HNSW
    index = hnswlib.Index(space='cosine', dim=384)
    index.load_index('models/hnsw_index.bin')
    index.set_ef(50)

    loaded = True

# Limpeza de texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

# Função auxiliar Word2Vec
def w2v_embed(text):
    # Lista simples de stop words em inglês
    stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'with', 'about', 'to', 'of', 'movie', 'film'])
    
    words_with_bigrams = bigram_phraser[text.split()]
    # Filtra palavras que estão no vocabulário e não são stop words
    words = [w for w in words_with_bigrams if w in w2v_model.wv and w not in stop_words]
    if not words:
        # Se não sobrar palavras, retorna um vetor nulo
        return np.zeros(100)
    return np.mean([w2v_model.wv[w] for w in words], axis=0)

# Função auxiliar para Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(results_lists, k=60):
    """
    Combina múltiplas listas de resultados classificados usando RRF.
    Args:
        results_lists: Uma lista de listas, onde cada lista interna contém IDs de documentos.
        k: Constante para estabilizar a pontuação.
    Returns:
        Um dicionário mapeando o ID do documento para sua pontuação RRF.
    """
    fused_scores = {}
    for results in results_lists:
        for rank, doc_id in enumerate(results):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1) # +1 para rank baseado em 1
    
    return fused_scores

# Função principal de busca
def search(query, top_k=5, method='hnsw'):

    load_models()

    if method == 'hnsw':
        # Geração de candidatos
        # Gera candidatos de 3 fontes para maior robustez.

        # Busca Semântica na Query Original (captura a intenção direta)
        original_query_emb = sbert_model.encode([query])
        original_hnsw_labels, _ = index.knn_query(original_query_emb, k=top_k * 4)
        original_hnsw_candidates = original_hnsw_labels[0]

        #  Busca Semântica com HyDE (expansão conceitual)
        hyde_prompt = f"""You are a movie expert. Your task is to write a hypothetical movie summary that perfectly answers the user's question.
Focus on expanding the user's query into a plausible summary. **It is crucial to incorporate the key terms and concepts from the user's question directly into the summary you generate.**

User Question: {query}

Hypothetical Summary:"""
        output = llm(hyde_prompt, max_new_tokens=100)[0]['generated_text']
        hypothetical_doc = output.split("Hypothetical Summary:")[-1].strip()
        print(f"Hypothetical Document for Search: {hypothetical_doc}")
        hyde_emb = sbert_model.encode([hypothetical_doc])
        hyde_hnsw_labels, _ = index.knn_query(hyde_emb, k=top_k * 4)
        hyde_hnsw_candidates = hyde_hnsw_labels[0]

        #  Busca por Palavra-chave (TF-IDF)
        query_vec = tfidf.transform([clean_text(query)])
        tfidf_sims = cosine_similarity(query_vec, tfidf_matrix)[0]
        tfidf_candidates = tfidf_sims.argsort()[-(top_k * 4):][::-1] # Obter mais candidatos

        # RECLASSIFICAÇÃO COM RRF 
        fused_scores = reciprocal_rank_fusion([original_hnsw_candidates, hyde_hnsw_candidates, tfidf_candidates])
        reranked_ids = sorted(fused_scores.keys(), key=lambda id: fused_scores[id], reverse=True)
        
        top_idx = reranked_ids[:top_k]
        results = df.iloc[top_idx][['title', 'summary']].copy()
        max_score = fused_scores[top_idx[0]] if top_idx else 0
        results['score'] = [fused_scores[i] / max_score if max_score > 0 else 0 for i in top_idx]

    elif method == 'w2v':
        query_emb = w2v_embed(clean_text(query)).reshape(1, -1)
        sims = cosine_similarity(query_emb, w2v_embeddings)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        results = df.iloc[top_idx][['title', 'summary']].copy()
        results['score'] = sims[top_idx]

    elif method == 'tfidf':
        query_vec = tfidf.transform([clean_text(query)])
        sims = cosine_similarity(query_vec, tfidf_matrix)[0]
        top_idx = sims.argsort()[-top_k:][::-1]
        results = df.iloc[top_idx][['title', 'summary']].copy()
        results['score'] = sims[top_idx]

    else:
        raise ValueError(f"Método inválido: {method}. Use 'hnsw', 'w2v' ou 'tfidf'.")

    return results.reset_index(drop=True)