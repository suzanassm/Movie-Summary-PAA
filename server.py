from flask import Flask, request, jsonify, render_template
from search import search
from llm_response import generate_answer

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json

    if not data or 'query' not in data:
        return jsonify({'error': 'Campo "query" é obrigatório.'}), 400

    query = data.get('query', '')
    method = data.get('method', 'hnsw')
    top_k = data.get('top_k', 5)

    try:
        results = search(query, top_k=top_k, method=method)
        answer = generate_answer(query, results)

        return jsonify({
            'query': query,
            'method': method,
            'answer': answer,
            'movies_found': results['title'].tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search_only():
    """Busca sem LLM — retorna apenas os filmes encontrados."""
    data = request.json

    if not data or 'query' not in data:
        return jsonify({'error': 'Campo "query" é obrigatório.'}), 400

    query = data.get('query', '')
    method = data.get('method', 'hnsw')
    top_k = data.get('top_k', 5)

    try:
        results = search(query, top_k=top_k, method=method)
        return jsonify({
            'query': query,
            'method': method,
            'results': results['title'].to_list()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)