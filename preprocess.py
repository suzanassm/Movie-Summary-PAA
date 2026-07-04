import pandas as pd
import re
import os

os.makedirs('data', exist_ok=True)

# Carrega metadados
movies = pd.read_csv('data/movie.metadata.tsv', sep='\t', header=None,
    usecols=[0, 2], names=['movie_id', 'title'])

# Carrega sinopses
summaries = pd.read_csv('data/plot_summaries.txt', sep='\t', header=None,
    names=['movie_id', 'summary'])

# Une
df = movies.merge(summaries, on='movie_id').dropna()

# Limpeza de texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df['summary_clean'] = df['summary'].apply(clean_text)

# Cria um campo combinado para busca, dando mais peso ao título
df['text_for_search'] = df['title'].str.lower() + ' ' + df['summary_clean']

# Salva
df.to_csv('data/processed.csv', index=False)
print(f"Pré-processamento concluído! {len(df)} filmes carregados.")