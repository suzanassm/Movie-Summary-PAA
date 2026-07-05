from transformers import pipeline

# Carregar modelo LLM local (TinyLlama)
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device=-1  # CPU; trocar para 0 se tiver GPU
)


def generate_answer(query, results):
    context = "\n\n".join([
        f"Movie: {row['title']}\nSummary: {row['summary'][:300]}"
        for _, row in results.iterrows()
    ])

    # TinyLlama-Chat foi ajustado (fine-tuned) para seguir o formato de chat
    # com papéis explícitos (system/user/assistant). Mandar um prompt como
    # texto corrido, sem esses papéis, faz o modelo tratar tudo como um bloco
    # de texto para "continuar" em vez de uma instrução para seguir — e é
    # comum ele reagir ecoando de volta trechos do próprio prompt.
    messages = [
        {
            "role": "system",
            "content": "You are a movie expert assistant. Based ONLY on the movie "
                       "summaries provided below, answer the user's question clearly "
                       "and helpfully, in a couple of sentences. Do not add plot "
                       "details, actor names, or facts that are not explicitly "
                       "present in the summaries given to you. If the summaries "
                       "don't clearly answer the question, say so honestly instead "
                       "of inventing information.",
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nMovies found:\n{context}",
        },
    ]

    # apply_chat_template monta o prompt com os tokens especiais corretos
    # (<|system|>, <|user|>, <|assistant|>) que o TinyLlama-Chat espera.
    prompt = llm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    output = llm(
        prompt,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        repetition_penalty=1.2,  # reduz a chance de repetir/ecoar o próprio prompt
        return_full_text=False,  # retorna só o texto gerado, sem o prompt de entrada
    )

    answer = output[0]['generated_text'].strip()

    # Salvaguarda: se por algum motivo o modelo ainda assim ecoar o prompt
    # (ex: gerar algo terminando prematuramente), evita mostrar isso ao usuário.
    if not answer or answer.lower().startswith("you are a movie expert"):
        answer = ("Não consegui gerar uma resposta clara para essa pergunta "
                   "com os filmes encontrados. Veja a lista de filmes abaixo.")

    return answer