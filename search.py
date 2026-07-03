import numpy as np
import pandas as pd
import pickle
import re
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import hnswlib

loaded = False

def load_models():
    global loaded, df, w2v_model, sbert_model, tfidf, tfidf_matrix
    global w2v_embeddings, index

    if loaded:
        return

    # Carregar dados
    df = pd.read_csv('data/processed.csv')

    # Carregar modelos
    w2v_model = Word2Vec.load('models/w2v_model.bin')
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    with open('models/tfidf.pkl', 'rb') as f:
        tfidf, tfidf_matrix = pickle.load(f)

    #w2v_embeddings = np.load('models/w2v_embeddings.npy')
    w2v_embeddings = np.load('models/w2v_embeddings.npy', mmap_mode='r')

    # Carregar índice HNSW
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
    words = [w for w in text.split() if w in w2v_model.wv]
    if not words:
        return np.zeros(100)
    return np.mean([w2v_model.wv[w] for w in words], axis=0)

# Função principal de busca
def search(query, top_k=5, method='hnsw'):

    load_models()

    if method == 'hnsw':
        query_emb = sbert_model.encode([query])
        labels, distances = index.knn_query(query_emb, k=top_k)
        results = df.iloc[labels[0]][['title', 'summary']].copy()
        results['score'] = 1 - distances[0]

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