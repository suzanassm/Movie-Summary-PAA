import pandas as pd
import re
import os
import shutil

os.makedirs("data", exist_ok=True)

if os.path.exists("MovieSummaries"):

    destino = os.path.join("data", "MovieSummaries")

    if os.path.exists(destino):
        shutil.rmtree(destino)   # remove a versão antiga

    shutil.move("MovieSummaries", destino)

if os.path.exists("MovieSummaries.tar.gz"):
    os.remove("MovieSummaries.tar.gz")
# Carrega metadados
movies = pd.read_csv('data/MovieSummaries/movie.metadata.tsv', sep='\t', header=None,
                      usecols=[0, 2], names=['movie_id', 'title'])

# Carrega sinopses
summaries = pd.read_csv('data/MovieSummaries/plot_summaries.txt', sep='\t', header=None,
                         names=['movie_id', 'summary'])

# Une
df = movies.merge(summaries, on='movie_id').dropna()

# Limpeza de texto
# IMPORTANTE: esta função precisa ficar IDÊNTICA à clean_text usada em search.py,
# senão o vocabulário treinado não bate com a tokenização feita na hora da busca.
def clean_text(text):
    text = text.lower()
    text = text.replace('-', ' ')  # "spider-man" -> "spider man" (mesma regra do search.py)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df['summary_clean'] = df['summary'].apply(clean_text)

# Cria um campo combinado para busca, dando mais peso ao título
# (aplica a mesma limpeza no título, em vez de só lower(), para manter consistência)
df['text_for_search'] = df['title'].apply(clean_text) + ' ' + df['summary_clean']

# Salva
df.to_csv('data/processed.csv', index=False)
print(f"Pré-processamento concluído! {len(df)} filmes carregados.")