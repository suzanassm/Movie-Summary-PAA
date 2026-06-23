from transformers import pipeline

# Carregar modelo LLM local (TinyLlama)
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=300,
    device=-1  # CPU; trocar para 0 se tiver GPU
)

def generate_answer(query, results):
    context = "\n\n".join([
        f"Movie: {row['title']}\nSummary: {row['summary'][:300]}"
        for _, row in results.iterrows()
    ])

    prompt = f"""You are a movie expert assistant.
Based on the summaries below, answer the user's question clearly and helpfully.

Question: {query}

Movies found:
{context}

Answer:"""

    output = llm(prompt)[0]['generated_text']
    answer = output.split("Answer:")[-1].strip()
    return answer