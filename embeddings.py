import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import hnswlib
import pickle
import os

# Carregar dados processados
df = pd.read_csv('data/processed.csv')

# --- Word2Vec ---
sentences = [text.split() for text in df['summary_clean']]
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
os.makedirs("models", exist_ok=True)
w2v_model.save('models/w2v_model.bin')

# Gerar e salvar embeddings Word2Vec (faltava isso!)
def w2v_embed(text):
    words = [w for w in text.split() if w in w2v_model.wv]
    if not words:
        return np.zeros(100)
    return np.mean([w2v_model.wv[w] for w in words], axis=0)

w2v_embeddings = np.array([w2v_embed(t) for t in df['summary_clean']])
np.save('models/w2v_embeddings.npy', w2v_embeddings)
print("Word2Vec embeddings gerados!")

# --- TF-IDF ---
tfidf = TfidfVectorizer(max_features=10000)
tfidf_matrix = tfidf.fit_transform(df['summary_clean'])
with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump((tfidf, tfidf_matrix), f)
print("TF-IDF gerado!")

# --- SBERT ---
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
sbert_embeddings = sbert_model.encode(df['summary'].tolist(), batch_size=64, show_progress_bar=True)
np.save('models/sbert_embeddings.npy', sbert_embeddings)
print("SBERT embeddings gerados!")

# --- HNSW (construído sobre os embeddings SBERT) ---
dim = sbert_embeddings.shape[1]
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=len(sbert_embeddings), ef_construction=200, M=16)
index.add_items(sbert_embeddings, list(range(len(sbert_embeddings))))
index.set_ef(50)
index.save_index('models/hnsw_index.bin')
print("Índice HNSW gerado!")

print("\nEmbeddings e índice HNSW gerados com sucesso!")