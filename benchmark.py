import time
from search import search

queries_teste = [
    "a superhero saves the city from destruction",
    "love story in paris between two strangers",
    "science fiction space travel to another galaxy",
    "a boy discovers he has magical powers",
    "crime thriller with a detective solving a murder"
]

methods = ['tfidf', 'w2v', 'hnsw']

print("=" * 60)
print("BENCHMARK DE MÉTODOS DE BUSCA SEMÂNTICA")
print("=" * 60)

for method in methods:
    tempos = []
    for query in queries_teste:
        inicio = time.time()
        results = search(query, top_k=5, method=method)
        tempos.append(time.time() - inicio)

    media = sum(tempos) / len(tempos)
    minimo = min(tempos)
    maximo = max(tempos)

    print(f"\nMétodo: {method.upper()}")
    print(f"  Tempo médio : {media:.4f}s")
    print(f"  Tempo mínimo: {minimo:.4f}s")
    print(f"  Tempo máximo: {maximo:.4f}s")

print("\n" + "=" * 60)
print("Exemplo de resultados para a query: 'space travel'")
print("=" * 60)

for method in methods:
    print(f"\n[{method.upper()}]")
    results = search("space travel", top_k=3, method=method)
    for _, row in results.iterrows():
        print(f"  - {row['title']} (score: {row['score']:.4f})")