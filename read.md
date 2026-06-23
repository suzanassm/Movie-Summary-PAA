# Movie Summary Search System

Trabalho desenvolvido na disciplina de Projeto e Análise de Algoritmos 26.1

## Sobre o projeto

O sistema recebe uma pergunta em linguagem natural sobre filmes e retorna resultados relevantes usando busca semântica no dataset CMU Movie Summary Corpus. A resposta final é formatada por um LLM local.

Foram implementados e comparados três métodos de busca semântica: TF-IDF, Word2Vec e Sentence Embeddings (SBERT) com índice HNSW. Dessa forma, avaliou-se qual oferece melhor equilíbrio entre qualidade e desempenho.

## Integrantes

| Nome | Matrícula |
|---|---|
| Bernardo Vilar | 232009487 |
| Fernando Lovato Weber | 232037641 |
| Isabela Soares Furlan | 231013636 |
| Lucas da Costa Rodrigues | 221017079 |
| Suzana Miranda | 231037020 |
| Wallysson Matheus de Queiroz Silva | 231038798 |

---

## Como rodar

### Requisitos

- Python 3.11 
- versões mais novas (3.12+) têm problema de compatibilidade com o gensim



### 1. Criar o ambiente virtual

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 3. Baixar o dataset

```bash
curl -O http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz
tar -xzf MovieSummaries.tar.gz
```

Mova os arquivos extraídos para a pasta `data/`. Os dois arquivos usados são `plot_summaries.txt` e `movie.metadata.tsv`.

### 4. Pré-processar os dados e gerar os embeddings

Rode nessa ordem:

```bash
python preprocess.py
python embeddings.py
```

### 5. Subir o servidor

```bash
python server.py
```

Acesse em `http://localhost:5000`.

### 6. Fazer uma consulta

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "a movie about a boy who discovers he is a wizard", "method": "hnsw"}'
```

O campo `method` aceita `hnsw`, `w2v` ou `tfidf`. Se omitido, usa `hnsw` por padrão.

Para buscar sem passar pelo LLM (só os filmes encontrados):
```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "space travel", "method": "tfidf", "top_k": 3}'
```

### 7. Comparar os métodos 

```bash
python benchmark.py
```

Roda um conjunto de queries nos três métodos e imprime o tempo médio de cada um.

---

## Estrutura

```
.
├── data/
│   ├── movie.metadata.tsv
│   └── plot_summaries.txt
├── models/
│   ├── w2v_model.bin
│   ├── w2v_embeddings.npy
│   ├── sbert_embeddings.npy
│   ├── hnsw_index.bin
│   └── tfidf.pkl
├── preprocess.py       # limpeza e merge dos dados
├── embeddings.py       # treina Word2Vec, SBERT, TF-IDF e HNSW
├── search.py           # funções de busca para cada método
├── llm_response.py     # monta o prompt e chama o TinyLlama
├── server.py           # API Flask com rotas /ask e /search
├── benchmark.py        # compara tempo de resposta dos métodos
├── requirements.txt
└── README.md
```